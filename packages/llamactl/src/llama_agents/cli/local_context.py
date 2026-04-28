# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Gather local context for deployment scaffolding.

Reads the current working directory: git remote / branch, deployment config,
``.env`` secrets, and the installed appserver version. Produces a
:class:`LocalContext` consumable by ``deployments template`` and (later) the
interactive editor.

Textual-free by design — both the CLI ``deployments template`` command and the
future Slice C2 editor depend on it. The Textual TUI keeps its own copy of the
gathering logic in ``textual/deployment_form.py`` until that screen is deleted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from llama_agents.cli.env import load_env_secrets_from_string
from llama_agents.cli.utils.version import get_installed_appserver_version
from llama_agents.core.deployment_config import (
    DEFAULT_DEPLOYMENT_NAME,
    read_deployment_config,
)
from llama_agents.core.git.git_util import (
    get_current_branch,
    get_git_root,
    is_git_repo,
    list_remotes,
)


@dataclass(frozen=True)
class LocalContext:
    """Local-machine signals used to scaffold a deployment YAML.

    All fields are passive: this is a snapshot of what we observed, not a
    decision about what the resulting deployment should look like.
    """

    is_git_repo: bool = False
    repo_url: str | None = None
    git_ref: str | None = None
    display_name: str | None = None
    deployment_file_path: str | None = None
    available_secrets: dict[str, str] = field(default_factory=dict)
    required_secret_names: list[str] = field(default_factory=list)
    installed_appserver_version: str | None = None
    warnings: list[str] = field(default_factory=list)


def gather_local_context() -> LocalContext:
    """Read cwd-rooted signals into a :class:`LocalContext`.

    Mirrors the behaviour of the TUI's ``_initialize_deployment_data`` minus
    push-mode/server-capability state, which belongs to the consumer. Failures
    while parsing the deployment config become a generic warning rather than
    an exception, matching the existing TUI behaviour.
    """

    warnings: list[str] = []
    display_name: str | None = None
    deployment_file_path: str | None = None
    required_secret_names: list[str] = []

    has_git = is_git_repo()
    try:
        config = read_deployment_config(Path("."), Path("."))
        if config.name != DEFAULT_DEPLOYMENT_NAME:
            display_name = config.name
        required_secret_names = list(config.required_env_vars)
    except Exception:
        warnings.append("Could not parse local deployment config. It may be invalid.")

    repo_url: str | None = None
    git_ref: str | None = None
    if has_git:
        seen: set[str] = set()
        candidates: list[str] = []
        for remote in list_remotes():
            normalized = _normalize_to_http(remote)
            if normalized not in seen:
                candidates.append(normalized)
                seen.add(normalized)
        preferred = sorted(candidates, key=lambda x: "github.com" in x, reverse=True)
        if preferred:
            repo_url = preferred[0]
        git_ref = get_current_branch()
        try:
            root = get_git_root()
            if root != Path.cwd():
                deployment_file_path = str(Path.cwd().relative_to(root))
        except Exception:
            pass

    available_secrets: dict[str, str] = {}
    env_path = Path(".env")
    if env_path.exists():
        try:
            available_secrets = load_env_secrets_from_string(env_path.read_text())
        except Exception:
            pass

    return LocalContext(
        is_git_repo=has_git,
        repo_url=repo_url,
        git_ref=git_ref,
        display_name=display_name,
        deployment_file_path=deployment_file_path,
        available_secrets=available_secrets,
        required_secret_names=required_secret_names,
        installed_appserver_version=get_installed_appserver_version(),
        warnings=warnings,
    )


def _normalize_to_http(url: str) -> str:
    """Best-effort normalize a git URL to an https URL.

    Handles the common SSH/SCP shapes (``git@host:path``,
    ``ssh://git@host:port/path``) and bare ``host/path`` strings. Strips
    credentials and any explicit port. Equivalent to the TUI's helper of the
    same name; lifted here so both consumers can share it.
    """
    candidate = (url or "").strip()

    has_scheme = "://" in candidate
    if not has_scheme:
        scp_match = re.match(
            r"^(?:(?P<user>[^@]+)@)?(?P<host>[^:/\s]+):(?P<path>[^/].+)$",
            candidate,
        )
        if scp_match:
            host = scp_match.group("host")
            path = scp_match.group("path").lstrip("/")
            if path.endswith(".git"):
                path = path[:-4]
            return f"https://{host}/{path}"

    parsed = urlsplit(candidate if has_scheme else f"https://{candidate}")
    netloc = parsed.netloc.split("@", 1)[-1]
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    path = parsed.path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if path:
        return f"https://{netloc}/{path}"
    return f"https://{netloc}"
