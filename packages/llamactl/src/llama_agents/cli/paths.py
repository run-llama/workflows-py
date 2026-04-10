# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import os
from pathlib import Path


def resolve_llamactl_config_dir() -> Path:
    """Resolve the config directory from the override/env/platform policy."""
    override = os.environ.get("LLAMACTL_CONFIG_DIR")
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        return (Path(os.environ.get("APPDATA", "~")) / "llamactl").expanduser()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return (Path(xdg_config_home).expanduser() / "llamactl").expanduser()
    return (Path.home() / ".config" / "llamactl").expanduser()


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
