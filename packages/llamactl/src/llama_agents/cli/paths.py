# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_config_path


def legacy_llamactl_config_dir() -> Path:
    """Return the historical llamactl config directory."""
    if os.name == "nt":
        return (Path(os.environ.get("APPDATA", "~")) / "llamactl").expanduser()
    return (Path.home() / ".config" / "llamactl").expanduser()


def standard_llamactl_config_dir() -> Path:
    """Return the platform-standard llamactl config directory."""
    return user_config_path("llamactl", appauthor=False)


def resolve_llamactl_config_dir() -> Path:
    """Resolve the config directory without migrating existing profile state."""
    override = os.environ.get("LLAMACTL_CONFIG_DIR")
    if override:
        return Path(override).expanduser()

    legacy = legacy_llamactl_config_dir()
    standard = standard_llamactl_config_dir()
    if legacy == standard:
        return standard
    if (legacy / "profiles.db").exists():
        return legacy
    return standard


def bash_completion_dir(home: Path | None = None) -> Path:
    """Return the preferred per-user bash completion directory."""
    resolved_home = home or Path.home()
    preferred = resolved_home / ".local" / "share" / "bash-completion" / "completions"
    if preferred.exists():
        return preferred
    return resolved_home / ".bash_completion.d"


def bash_rc_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".bashrc"


def zsh_completion_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".zfunc"


def zsh_rc_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".zshrc"


def fish_completion_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".config" / "fish" / "completions"
