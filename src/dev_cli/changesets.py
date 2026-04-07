"""
This is a script called by the changeset bot. Normally changeset can do the following things, but this is a mixed ts and python repo, so we need to do some extra things.

There's 2 things this does:
- Versioning: Makes changes that may be committed with the newest version.
- Releasing/Tagging: After versions are changed, we check each package to see if its released, and if not, we release it and tag it.

"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from io import StringIO
from pathlib import Path
from typing import Any, Generator, List, cast

import click
import tomlkit
from packaging.version import Version
from pydantic import BaseModel, Field
from ruamel.yaml import YAML

# Valid PEP 440 pre-release labels. Semver pre-release identifiers must use
# these same labels (e.g. 1.2.3-a.4, not 1.2.3-alpha.4) so that the
# conversion is purely structural.
_PEP440_LABELS = {"a", "b", "rc"}

_SEMVER_PRERELEASE_RE = re.compile(r"^(\d+\.\d+\.\d+)-([a-zA-Z]+)\.(\d+)$")


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


def semver_to_pep440(version: str) -> str:
    """Convert a semver version string to PEP 440 format.

    Only PEP 440-compatible pre-release labels are accepted (a, b, rc):
        1.2.3-a.4   -> 1.2.3a4
        1.2.3-b.1   -> 1.2.3b1
        1.2.3-rc.2  -> 1.2.3rc2

    Non-prerelease versions pass through unchanged.
    """
    match = _SEMVER_PRERELEASE_RE.match(version)
    if not match:
        return version

    base, label, num = match.groups()
    if label not in _PEP440_LABELS:
        raise ValueError(
            f"Unsupported pre-release label '{label}' in version '{version}'. "
            f"Use a PEP 440 label: {', '.join(sorted(_PEP440_LABELS))}"
        )
    return f"{base}{label}{num}"


def pep440_to_semver(version: str) -> str:
    """Convert a PEP 440 version string to semver format.

    Pre-release versions are converted:
        1.2.3a4   -> 1.2.3-a.4
        1.2.3b1   -> 1.2.3-b.1
        1.2.3rc2  -> 1.2.3-rc.2

    Non-prerelease versions pass through unchanged.
    """
    v = Version(version)
    base = ".".join(str(x) for x in v.release)
    if v.pre is None:
        return base

    label, num = v.pre
    return f"{base}-{label}.{num}"


class DockerConfig(BaseModel):
    dockerfile: str
    imageName: str
    target: str | None = None
    platforms: list[str] = Field(default_factory=list)


class HelmConfig(BaseModel):
    registry: str


# syncValues type: filename -> {dot_path -> template_string}
SyncValues = dict[str, dict[str, str]]


class PublishConfig(BaseModel):
    """Per-type publish toggles.  All default to ``True``; set to ``False``
    to suppress a specific publish channel even when the corresponding
    config (docker/helm/pyproject.toml) exists."""

    pypi: bool = True
    docker: bool = True
    helm: bool = True


class PackageJsonFile(BaseModel):
    """Schema for a package.json file on disk."""

    name: str
    docker: DockerConfig | None = None
    helm: HelmConfig | None = None
    publish: PublishConfig = Field(default_factory=PublishConfig)
    syncValues: dict[str, dict[str, str]] = Field(default_factory=dict)
    postVersion: list[str] = Field(default_factory=list)


@dataclass
class PackageJson:
    name: str
    version: str
    path: Path
    private: bool
    docker: DockerConfig | None = None
    helm: HelmConfig | None = None
    publish: PublishConfig = dataclasses_field(default_factory=PublishConfig)
    syncValues: SyncValues = dataclasses_field(default_factory=dict)
    postVersion: list[str] = dataclasses_field(default_factory=list)

    def should_publish_pypi(self) -> bool:
        return not self.private and self.publish.pypi

    def should_publish_docker(self) -> bool:
        return not self.private and self.publish.docker

    def should_publish_helm(self) -> bool:
        return not self.private and self.publish.helm


# --- syncValues engine ---

# Template pattern: {package-name:property} or {self:property}
_TEMPLATE_RE = re.compile(r"\{([^}:]+):([^}]+)\}")


def _resolve_template(
    template: str,
    self_pkg: PackageJson,
    workspace_packages: dict[str, PackageJson],
) -> str:
    """Resolve a syncValues template string like ``{pkg:dockerTag}``."""

    def _replace(match: re.Match[str]) -> str:
        pkg_ref, prop = match.group(1), match.group(2)
        template_str = f"{{{pkg_ref}:{prop}}}"
        if pkg_ref == "self":
            pkg = self_pkg
        else:
            pkg = workspace_packages.get(pkg_ref)
            if pkg is None:
                raise ValueError(
                    f"syncValues template '{template_str}' references "
                    f"unknown package '{pkg_ref}'"
                )
        if prop == "version":
            return pkg.version
        if prop == "pep440Version":
            return semver_to_pep440(pkg.version)
        if prop == "dockerTag":
            if pkg.docker is None:
                raise ValueError(
                    f"syncValues template '{template_str}' references "
                    f"dockerTag but package '{pkg.name}' has no docker config"
                )
            return pkg.version
        raise ValueError(
            f"syncValues template '{template_str}' uses unknown property '{prop}'"
        )

    return _TEMPLATE_RE.sub(_replace, template)


def _apply_dot_path_updates(data: Any, updates: dict[str, str]) -> bool:
    """Walk dot-paths and set values in a nested dict-like structure.

    Returns True if any value changed.
    """
    changed = False
    for dot_path, value in updates.items():
        keys = dot_path.split(".")
        obj = data
        for key in keys[:-1]:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]
        last_key = keys[-1]
        old = obj.get(last_key)
        if old is None or str(old) != value:
            obj[last_key] = value
            changed = True
    return changed


def _write_yaml_values(file_path: Path, updates: dict[str, str]) -> bool:
    """Set dot-path values in a YAML file using ruamel.yaml (comment-preserving).

    Returns True if the file was changed.
    """
    if not file_path.exists():
        return False

    yaml = YAML()
    yaml.preserve_quotes = True  # type: ignore[assignment]
    data = yaml.load(file_path.read_text())

    if not _apply_dot_path_updates(data, updates):
        return False

    stream = StringIO()
    yaml.dump(data, stream)
    file_path.write_text(stream.getvalue())
    return True


def _write_toml_values(file_path: Path, updates: dict[str, str]) -> bool:
    """Set dot-path values in a TOML file using tomlkit (comment-preserving).

    Returns True if the file was changed.
    """
    if not file_path.exists():
        return False

    doc = tomlkit.parse(file_path.read_text())

    if not _apply_dot_path_updates(doc, updates):
        return False

    file_path.write_text(tomlkit.dumps(doc))
    return True


def _write_file_values(file_path: Path, updates: dict[str, str]) -> bool:
    """Dispatch to the correct writer based on file extension."""
    suffix = file_path.suffix
    if suffix in (".yaml", ".yml"):
        return _write_yaml_values(file_path, updates)
    if suffix == ".toml":
        return _write_toml_values(file_path, updates)
    raise ValueError(f"Unsupported file type '{suffix}' for syncValues: {file_path}")


def apply_sync_values(
    pkg: PackageJson,
    workspace_packages: dict[str, PackageJson],
) -> bool:
    """Apply all sync entries (explicit + implicit) for a package.

    Explicit entries come from ``syncValues`` in package.json. Implicit
    entries are added for helm packages (Chart.yaml version) and packages
    with a pyproject.toml (project.version in PEP 440 form). Writers
    skip gracefully when the target file doesn't exist.

    Returns True if any file was changed.
    """
    all_entries: dict[str, dict[str, str]] = {}

    for filename, paths in pkg.syncValues.items():
        resolved = {
            dp: _resolve_template(tpl, pkg, workspace_packages)
            for dp, tpl in paths.items()
        }
        all_entries.setdefault(filename, {}).update(resolved)

    if pkg.helm is not None:
        all_entries.setdefault("Chart.yaml", {}).setdefault("version", pkg.version)

    if (pkg.path / "pyproject.toml").exists():
        all_entries.setdefault("pyproject.toml", {}).setdefault(
            "project.version", semver_to_pep440(pkg.version)
        )

    changed = False
    for filename, updates in all_entries.items():
        file_path = pkg.path / filename
        if _write_file_values(file_path, updates):
            click.echo(f"Updated {file_path}")
            changed = True

    return changed


def _read_package_json_config(package_dir: Path) -> PackageJsonFile | None:
    """Parse custom fields from a package.json file using Pydantic."""
    package_json_path = package_dir / "package.json"
    if not package_json_path.exists():
        return None
    data = json.loads(package_json_path.read_text())
    return PackageJsonFile.model_validate(data)


def get_pnpm_workspace_packages() -> list[PackageJson]:
    """Return directories for all workspace packages from pnpm list JSON output."""
    output = run_and_capture(["pnpm", "list", "-r", "--depth=-1", "--json"])

    package_json = cast(list[dict[str, Any]], json.loads(output))
    packages: list[PackageJson] = []
    for data in package_json:
        pkg_path = Path(data["path"])
        config = _read_package_json_config(pkg_path)
        packages.append(
            PackageJson(
                name=data["name"],
                version=data["version"],
                path=pkg_path,
                private=data.get("private", True),
                docker=config.docker if config else None,
                helm=config.helm if config else None,
                publish=config.publish if config else PublishConfig(),
                syncValues=config.syncValues if config else {},
                postVersion=config.postVersion if config else [],
            )
        )
    return packages


def _pypi_packages(packages: list[PackageJson]) -> Generator[Path, None, None]:
    """Yield pyproject.toml paths for packages that should be published to PyPI."""
    for package in packages:
        if not package.should_publish_pypi():
            continue
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


def maybe_publish_pypi(dry_run: bool, packages: list[PackageJson]) -> None:
    """Publish the py packages if they need to be published.

    Falls back to PyPI trusted publishing when UV_PUBLISH_TOKEN is not set.
    """
    token = os.environ.get("UV_PUBLISH_TOKEN")
    auth_method = "token" if token else "trusted publishing"
    click.echo(f"PyPI auth: {auth_method}")

    any_to_publish = False
    for package in _pypi_packages(packages):
        name, version = current_version(package)
        if is_published(name, version):
            click.echo(f"PyPI package {name}@{version} already published, skipping")
            continue
        any_to_publish = True
        click.echo(f"Publishing PyPI package {name}@{version}")

        if dry_run:
            click.echo("  uv build (dry run, skipping)")
        else:
            run_command(["uv", "build"], cwd=package.parent)
    if any_to_publish:
        if dry_run:
            click.echo("Dry run, skipping publish. Would run:")
            click.echo("  uv publish")
        else:
            publish_env = {**os.environ, "UV_PUBLISH_TOKEN": token} if token else None
            run_command(["uv", "publish"], env=publish_env)


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


DOCKER_REGISTRY = "docker.io"


def is_rc_version(version: str) -> bool:
    """Return True if version is a pre-release (RC, alpha, or beta)."""
    return bool(re.search(r"(-rc|-a|-b|rc\d|a\d|b\d)", version))


def is_docker_image_published(repository: str, tag: str) -> bool:
    """Check if a Docker image tag exists on Docker Hub.

    Returns True if the tag exists, False if not (404).
    Raises on unexpected HTTP errors.
    """
    url = f"https://hub.docker.com/v2/repositories/{repository}/tags/{tag}"
    try:
        urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    return True


def docker_image_tags(image: DockerConfig, version: str, is_rc: bool) -> list[str]:
    """Generate the full list of Docker tags for an image."""
    repo = f"{DOCKER_REGISTRY}/{image.imageName}"
    tags = [f"{repo}:{version}"]
    if not is_rc:
        tags.append(f"{repo}:latest")
        major_minor = ".".join(version.split(".")[:2])
        tags.append(f"{repo}:{major_minor}")
    return tags


def build_and_push_docker_image(image: DockerConfig, tags: list[str]) -> None:
    """Build and push a Docker image using docker buildx."""
    cmd = [
        "docker",
        "buildx",
        "build",
        "--push",
        "--file",
        image.dockerfile,
        "--platform",
        ",".join(image.platforms),
    ]
    if image.target:
        cmd.extend(["--target", image.target])
    for tag in tags:
        cmd.extend(["--tag", tag])
    cmd.append(".")
    run_command(cmd)


def maybe_publish_docker(dry_run: bool, packages: list[PackageJson]) -> None:
    """Build and push Docker images that haven't been published yet."""
    if not shutil.which("docker"):
        click.echo("Warning: docker not found in PATH, skipping Docker builds")
        return

    for pkg in packages:
        image = pkg.docker
        if image is None or not pkg.should_publish_docker():
            continue
        version = pkg.version
        rc = is_rc_version(version)
        primary_tag = version
        if is_docker_image_published(image.imageName, primary_tag):
            click.echo(
                f"Docker image {image.imageName}:{primary_tag} already published, skipping"
            )
            continue

        tags = docker_image_tags(image, version, rc)
        click.echo(f"Building and pushing Docker image {pkg.name}: {tags}")
        if dry_run:
            click.echo(f"  Dry run, skipping build for {pkg.name}")
        else:
            build_and_push_docker_image(image, tags)


def is_helm_chart_published(chart_name: str, version: str) -> bool:
    """Check if a Helm chart version is already published in the OCI registry."""
    url = (
        f"https://hub.docker.com/v2/repositories/llamaindex/{chart_name}/tags/{version}"
    )
    try:
        urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    return True


def maybe_publish_helm(dry_run: bool, packages: list[PackageJson]) -> None:
    """Package and push Helm charts that haven't been published yet."""
    if not shutil.which("helm"):
        click.echo("Warning: helm not found in PATH, skipping Helm chart publish")
        return

    for pkg in packages:
        if pkg.helm is None or not pkg.should_publish_helm():
            continue

        if is_helm_chart_published(pkg.name, pkg.version):
            click.echo(
                f"Helm chart {pkg.name}@{pkg.version} already published, skipping"
            )
            continue

        chart_dir = str(pkg.path)
        tgz = f"{pkg.name}-{pkg.version}.tgz"

        click.echo(f"Publishing Helm chart {tgz}")
        if dry_run:
            click.echo(f"  Dry run, would run: helm package {chart_dir}")
            click.echo(f"  Dry run, would run: helm push {tgz} {pkg.helm.registry}")
        else:
            run_command(["helm", "package", chart_dir])
            run_command(["helm", "push", tgz, pkg.helm.registry])


# ---------------------------------------------------------------------------
# Publish plan: a declarative description of the work to be done by
# ``changeset-publish``. The plan is emitted by ``changeset-plan`` and
# consumed by the per-action ``publish-*`` commands. The CI workflow
# fans these actions out into a GitHub Actions matrix so docker builds
# run natively (amd64 / arm64) in parallel.
# ---------------------------------------------------------------------------


@dataclass
class PypiAction:
    package: str
    version: str
    path: str  # package dir (absolute string)


@dataclass
class DockerBuildAction:
    package: str
    image: str  # imageName without registry
    dockerfile: str
    target: str | None
    platform: str  # single platform, e.g. "linux/amd64"
    version: str
    build_tag: str  # full registry/repo:version-<arch>
    cache_scope: str  # GHA buildx cache scope (per package + arch)


@dataclass
class DockerManifestAction:
    package: str
    image: str
    version: str
    final_tags: list[str]  # full tags that should resolve to the manifest
    source_tags: list[str]  # per-arch tags produced by docker-build jobs


@dataclass
class HelmAction:
    package: str
    chart_path: str
    version: str
    registry: str


@dataclass
class PublishPlan:
    pypi: list[PypiAction] = dataclasses_field(default_factory=list)
    docker_builds: list[DockerBuildAction] = dataclasses_field(default_factory=list)
    docker_manifests: list[DockerManifestAction] = dataclasses_field(
        default_factory=list
    )
    helm: list[HelmAction] = dataclasses_field(default_factory=list)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        from dataclasses import asdict

        return {
            "pypi": [asdict(a) for a in self.pypi],
            "docker_builds": [asdict(a) for a in self.docker_builds],
            "docker_manifests": [asdict(a) for a in self.docker_manifests],
            "helm": [asdict(a) for a in self.helm],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublishPlan:
        return cls(
            pypi=[PypiAction(**a) for a in data.get("pypi", [])],
            docker_builds=[
                DockerBuildAction(**a) for a in data.get("docker_builds", [])
            ],
            docker_manifests=[
                DockerManifestAction(**a) for a in data.get("docker_manifests", [])
            ],
            helm=[HelmAction(**a) for a in data.get("helm", [])],
        )


def _platform_suffix(platform: str) -> str:
    """Map a docker platform string to a short tag suffix."""
    # "linux/amd64" -> "amd64", "linux/arm64/v8" -> "arm64"
    parts = platform.split("/")
    if len(parts) >= 2:
        return parts[1]
    return platform.replace("/", "-")


def plan_pypi(packages: list[PackageJson]) -> list[PypiAction]:
    """Return PyPI actions for packages whose version is not yet published."""
    actions: list[PypiAction] = []
    repo_root = Path.cwd()
    for pyproject in _pypi_packages(packages):
        name, version = current_version(pyproject)
        if is_published(name, version):
            continue
        try:
            rel = pyproject.parent.relative_to(repo_root)
            path_str = str(rel)
        except ValueError:
            path_str = str(pyproject.parent)
        actions.append(PypiAction(package=name, version=version, path=path_str))
    return actions


def plan_docker(
    packages: list[PackageJson],
) -> tuple[list[DockerBuildAction], list[DockerManifestAction]]:
    """Return (per-arch build actions, manifest-merge actions) for unpublished images."""
    builds: list[DockerBuildAction] = []
    manifests: list[DockerManifestAction] = []

    for pkg in packages:
        image = pkg.docker
        if image is None or not pkg.should_publish_docker():
            continue
        version = pkg.version
        if is_docker_image_published(image.imageName, version):
            continue

        rc = is_rc_version(version)
        final_tags = docker_image_tags(image, version, rc)
        repo = f"{DOCKER_REGISTRY}/{image.imageName}"
        source_tags: list[str] = []
        for platform in image.platforms:
            suffix = _platform_suffix(platform)
            build_tag = f"{repo}:{version}-{suffix}"
            source_tags.append(build_tag)
            builds.append(
                DockerBuildAction(
                    package=pkg.name,
                    image=image.imageName,
                    dockerfile=image.dockerfile,
                    target=image.target,
                    platform=platform,
                    version=version,
                    build_tag=build_tag,
                    cache_scope=f"{pkg.name}-{suffix}",
                )
            )
        manifests.append(
            DockerManifestAction(
                package=pkg.name,
                image=image.imageName,
                version=version,
                final_tags=final_tags,
                source_tags=source_tags,
            )
        )

    return builds, manifests


def plan_helm(packages: list[PackageJson]) -> list[HelmAction]:
    """Return Helm chart actions for charts not yet pushed to the registry."""
    actions: list[HelmAction] = []
    repo_root = Path.cwd()
    for pkg in packages:
        if pkg.helm is None or not pkg.should_publish_helm():
            continue
        if is_helm_chart_published(pkg.name, pkg.version):
            continue
        try:
            chart_path = str(pkg.path.relative_to(repo_root))
        except ValueError:
            chart_path = str(pkg.path)
        actions.append(
            HelmAction(
                package=pkg.name,
                chart_path=chart_path,
                version=pkg.version,
                registry=pkg.helm.registry,
            )
        )
    return actions


def build_publish_plan(packages: list[PackageJson]) -> PublishPlan:
    """Scan workspace packages and produce a complete PublishPlan."""
    builds, manifests = plan_docker(packages)
    return PublishPlan(
        pypi=plan_pypi(packages),
        docker_builds=builds,
        docker_manifests=manifests,
        helm=plan_helm(packages),
    )


def execute_pypi_action(action: PypiAction, dry_run: bool = False) -> None:
    """Build and publish a single PyPI package."""
    click.echo(f"Publishing PyPI package {action.package}@{action.version}")
    if dry_run:
        click.echo("  dry run, skipping uv build / uv publish")
        return
    pkg_dir = Path(action.path)
    run_command(["uv", "build"], cwd=pkg_dir)
    token = os.environ.get("UV_PUBLISH_TOKEN")
    publish_env = {**os.environ, "UV_PUBLISH_TOKEN": token} if token else None
    # ``uv publish`` uploads everything in ./dist relative to cwd.
    run_command(["uv", "publish"], cwd=pkg_dir, env=publish_env)


def execute_docker_build_action(
    action: DockerBuildAction, dry_run: bool = False
) -> None:
    """Build and push a single-arch docker image for one platform."""
    click.echo(f"Building {action.image} ({action.platform}) -> {action.build_tag}")
    cmd = [
        "docker",
        "buildx",
        "build",
        "--push",
        "--file",
        action.dockerfile,
        "--platform",
        action.platform,
        "--tag",
        action.build_tag,
    ]
    if action.target:
        cmd.extend(["--target", action.target])
    # GHA buildx cache. Only emits cache directives when running under
    # GitHub Actions with the runtime token exposed (see the
    # ``crazy-max/ghaction-github-runtime`` step in the workflow). Outside
    # CI these env vars are absent and the flags are silently omitted so
    # local ``dev changeset-publish`` still works.
    if os.environ.get("ACTIONS_RUNTIME_TOKEN") and os.environ.get("ACTIONS_CACHE_URL"):
        cmd.extend(
            [
                "--cache-from",
                f"type=gha,scope={action.cache_scope}",
                "--cache-to",
                f"type=gha,mode=max,scope={action.cache_scope}",
            ]
        )
    cmd.append(".")
    if dry_run:
        click.echo(f"  dry run: {' '.join(cmd)}")
        return
    run_command(cmd)


def execute_docker_manifest_action(
    action: DockerManifestAction, dry_run: bool = False
) -> None:
    """Combine per-arch tags into a multi-arch manifest under each final tag."""
    click.echo(
        f"Creating manifest for {action.image}:{action.version} -> {action.final_tags}"
    )
    cmd = ["docker", "buildx", "imagetools", "create"]
    for tag in action.final_tags:
        cmd.extend(["--tag", tag])
    cmd.extend(action.source_tags)
    if dry_run:
        click.echo(f"  dry run: {' '.join(cmd)}")
        return
    run_command(cmd)


def execute_helm_action(action: HelmAction, dry_run: bool = False) -> None:
    """Package and push a single Helm chart."""
    tgz = f"{action.package}-{action.version}.tgz"
    click.echo(f"Publishing Helm chart {tgz} -> {action.registry}")
    if dry_run:
        click.echo(f"  dry run: helm package {action.chart_path}")
        click.echo(f"  dry run: helm push {tgz} {action.registry}")
        return
    run_command(["helm", "package", action.chart_path])
    run_command(["helm", "push", tgz, action.registry])


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
