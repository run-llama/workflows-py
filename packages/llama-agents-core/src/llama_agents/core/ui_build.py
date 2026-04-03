from __future__ import annotations

import json
from pathlib import Path

from .deployment_config import DeploymentConfig


def resolve_ui_root(config_parent: Path, config: DeploymentConfig) -> Path | None:
    """Return the absolute path to the UI root if UI is configured; otherwise None."""
    if config.ui is None:
        return None
    return (config_parent / config.ui.directory).resolve()


def ui_build_output_path(config_parent: Path, config: DeploymentConfig) -> Path | None:
    """
    Determine if the UI has a build script defined in package.json, and where the output will be.
    Right now, assumes its just `/dist` in the UI root.

    Returns:
    - Path to the build output directory if a package.json exists and contains a "build" script
    - None if there is no UI configured or no package.json exists
    """
    ui_root = resolve_ui_root(config_parent, config)
    if ui_root is None:
        return None
    if config.ui is None:
        return None
    package_json = ui_root / "package.json"
    if not package_json.exists():
        return None
    try:
        with open(package_json, "r", encoding="utf-8") as f:
            pkg = json.load(f)
        if not isinstance(pkg, dict):
            return None
        scripts = pkg.get("scripts", {}) or {}
        if config.ui.build_command in scripts:
            return config.build_output_path()
        return None
    except Exception:
        # Do not raise for malformed package.json in validation contexts
        return None
