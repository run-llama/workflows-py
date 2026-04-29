# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""YAML parsing for declarative ``llamactl deployments apply -f``.

Resolves ``${VAR}`` references against the process environment, strips the
``********`` mask passthrough convention used by ``get -o yaml``, preserves
explicit ``null`` secret values (semantic deletion), and validates the
result against :class:`DeploymentApply`.

Conventions:

* Top-level ``name:`` carries the stable deployment id (URL path on
  ``PUT /deployments/{name}``); the rest of the document is the apply body.
* ``${VAR}`` strings (in any field, not just secrets) resolve against the
  passed environment. Strict mode raises with all unset names listed.
* ``********`` secret values are passthroughs from ``get -o yaml`` and drop
  out of the body so a round-trip is a no-op.
* ``null`` secret values are preserved — server-side semantics delete them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import yaml
from llama_agents.core.schema.deployments import DeploymentApply
from pydantic import ValidationError

# Same value as ``llama_agents.cli.display.SECRET_MASK`` — kept local so this
# module can be imported without pulling the broader display graph.
_SECRET_MASK = "********"

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class UnresolvedVarsError(ValueError):
    """Raised when one or more ``${VAR}`` references cannot be resolved."""

    def __init__(self, names: list[str]) -> None:
        self.names = sorted(set(names))
        super().__init__("Unset environment variable(s): " + ", ".join(self.names))


class ApplyYamlError(ValueError):
    """Top-level error for malformed apply YAML (parse/validation)."""


@dataclass(frozen=True)
class ParsedApply:
    """Result of :func:`parse_apply_yaml`.

    ``name`` is the stable deployment id from the YAML — when present it
    becomes the URL path component of ``PUT /deployments/{name}``. ``apply``
    is the validated body. When ``name`` is ``None``, callers fall through to
    the create endpoint using ``apply.display_name``.
    """

    name: str | None
    apply: DeploymentApply


def resolve_env_vars(
    value: Any,
    *,
    env: Mapping[str, str] | None = None,
    strict: bool = True,
) -> tuple[Any, list[str]]:
    """Recursively replace ``${VAR}`` with ``env[VAR]`` in every string scalar.

    Returns ``(resolved, missing)``. In strict mode raises
    :class:`UnresolvedVarsError` when ``missing`` is non-empty. In non-strict
    mode unset references remain as the literal ``${VAR}`` string and are
    surfaced via the second return value for warning paths (the eventual
    editor flow). Walks dicts and lists recursively; non-string scalars pass
    through unchanged.
    """
    env_map = os.environ if env is None else env
    missing: list[str] = []

    def _walk(v: Any) -> Any:
        if isinstance(v, str):

            def _sub(m: re.Match[str]) -> str:
                name = m.group(1)
                if name in env_map:
                    return env_map[name]
                missing.append(name)
                return m.group(0)

            return _VAR_RE.sub(_sub, v)
        if isinstance(v, dict):
            return {k: _walk(item) for k, item in v.items()}
        if isinstance(v, list):
            return [_walk(item) for item in v]
        return v

    resolved = _walk(value)
    if strict and missing:
        raise UnresolvedVarsError(missing)
    return resolved, missing


def parse_apply_yaml(
    text: str,
    *,
    env: Mapping[str, str] | None = None,
    strict: bool = True,
    source: str | None = None,
) -> ParsedApply:
    """Parse declarative apply YAML into a :class:`ParsedApply`.

    Args:
        text: The YAML document body.
        env: Override for environment-variable resolution; defaults to
            ``os.environ``.
        strict: When True (default), unset ``${VAR}`` references raise
            :class:`UnresolvedVarsError`. When False, references stay literal.
        source: Optional file path / ``"-"`` annotation prepended to error
            messages so users can locate the offending document.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ApplyYamlError(
            _format_source(source, f"YAML parse error: {exc}")
        ) from exc

    if raw is None:
        raise ApplyYamlError(_format_source(source, "empty YAML document"))
    if not isinstance(raw, dict):
        raise ApplyYamlError(
            _format_source(
                source,
                f"YAML document must be a mapping, got {type(raw).__name__}",
            )
        )

    resolved, _ = resolve_env_vars(raw, env=env, strict=strict)

    name = resolved.pop("name", None)
    if name is not None and not isinstance(name, str):
        raise ApplyYamlError(_format_source(source, "'name' must be a string"))

    secrets = resolved.get("secrets")
    if isinstance(secrets, dict):
        # Drop ``********`` mask passthrough; preserve explicit ``None`` as
        # deletion. Non-string values fall through to schema validation.
        resolved["secrets"] = {k: v for k, v in secrets.items() if v != _SECRET_MASK}

    try:
        apply = DeploymentApply.model_validate(resolved)
    except ValidationError as exc:
        raise ApplyYamlError(_format_source(source, _format_validation(exc))) from exc

    return ParsedApply(name=name, apply=apply)


def _format_source(source: str | None, msg: str) -> str:
    return f"{source}: {msg}" if source else msg


def _format_validation(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "schema validation failed:\n  " + "\n  ".join(parts)
