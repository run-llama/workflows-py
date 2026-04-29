# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Render a :class:`DeploymentDisplay` as commented apply-shaped YAML.

One ``render()`` walks ``DeploymentSpec.model_fields`` in declaration order
and appends lines to a list. Each spec field renders in one of three states:

* **set** — ``<key>: <scalar>`` (uncommented), with the field's
  :class:`~llama_agents.cli.display.Doc` text as ``## …`` lines above.
  ``secrets`` is the only nested shape; its child keys render at indent 4.
* **required-but-unset** — ``<key>: ~`` with a ``## Required …`` marker so a
  ``grep`` or eyeball pass finds the gaps that block ``apply``.
* **unset** — ``# <key>: <example>`` (commented out one-liner) with the doc
  above. The schema-fixed ``_EXAMPLES`` table supplies the example value.

The top-level ``name`` follows the set / unset split (no ``required`` path —
the server slugifies an id when ``name`` is omitted, so a missing top-level
``name`` is never an apply blocker). The ``display_name`` spec field is a
special case: it always renders commented-out under its serialization alias
``generateName``, regardless of model value, since the canonical id is the
top-level ``name`` and ``generateName`` is opt-in slug-seed input.

A small ``field_alternatives`` mapping lets the caller surface a
commented-out alternative under a *set* field (used to suggest the detected
git remote under an empty ``repo_url``). Scalar quoting is a hand-rolled
plain-safe check; PyYAML handles the few values that need actual quoting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import yaml
from llama_agents.cli.display import DeploymentDisplay, DeploymentSpec, Doc
from pydantic.fields import FieldInfo

_INDENT = "  "
_MARKER = "## "
_REQUIRED = "Required — set before `apply`."

# Example value shown above the top-level ``name`` key when it is unset. The
# top-level ``name`` lives on ``DeploymentDisplay``, not ``DeploymentSpec`` —
# kept out of ``_EXAMPLES`` to avoid confusing the spec-field iteration.
_NAME_EXAMPLE = "my-app"

# Per-field example values shown when a spec field is unset (rendered commented).
# Keyed by the python field name on ``DeploymentSpec`` (matches
# ``model_fields`` iteration); the rendered YAML key may differ when the field
# has a serialization alias (e.g. ``display_name`` → ``generateName``).
_EXAMPLES: dict[str, Any] = {
    "display_name": "My App",
    # The push-mode sentinel (``""``) is documented separately via
    # :data:`~llama_agents.cli.display.PUSH_MODE_REPO_URL`; the example here
    # shows a real URL so a user uncommenting the line gets a working shape.
    "repo_url": "https://github.com/owner/repo",
    "deployment_file_path": ".",
    "git_ref": "main",
    "appserver_version": "0.5.0",
    "suspended": False,
    "secrets": {"MY_SECRET": "${MY_SECRET}"},
    "personal_access_token": "${GITHUB_TOKEN}",
}


def render(
    display: DeploymentDisplay,
    *,
    head: Sequence[str] = (),
    secret_comments: Mapping[str, str] = {},
    field_alternatives: Mapping[str, tuple[str, str]] = {},
    required: Iterable[str] = (),
) -> str:
    """Render ``display`` as a commented apply-shaped YAML string.

    Args:
        display: The deployment to render. ``status`` is unconditionally
            omitted; only ``name`` and ``spec`` reach the output.
        head: Lines emitted at the top of the output as ``## `` comments,
            before ``name:``. An empty-string entry becomes a bare ``##``.
        secret_comments: Map of secret name → comment text. Each becomes a
            ``## `` line above the matching key inside the ``secrets:``
            block. Ignored when ``secrets`` is unset.
        field_alternatives: Map of field name → ``(suggestion, annotation)``.
            For each *set* field listed, a commented-out
            ``# <field>: <suggestion>  # <annotation>`` line is emitted
            directly under the field. Silently ignored for fields in the
            ``required`` or ``unset`` states.
        required: Field names (python attribute names on ``DeploymentSpec``)
            to force-emit as ``<key>: ~`` with a ``## Required …`` marker
            even when unset. The top-level ``name`` is *not* supported here —
            it has no required path. ``display_name`` is silently ignored
            (always commented-out as ``generateName``).
    """
    required_set = set(required)
    out: list[str] = []

    for line in head:
        out.append("##" if line == "" else f"{_MARKER}{line}")

    name_docs = _docs(DeploymentDisplay.model_fields["name"])
    out.extend(_doc_lines(name_docs, indent=""))
    if display.name is None:
        out.append(f"# name: {_NAME_EXAMPLE}")
    else:
        out.append(f"name: {_scalar(display.name)}")

    out.append("spec:")
    # ``by_alias=True`` so ``display_name`` lands as ``generateName`` in the
    # rendered YAML. The dump is keyed by alias; we look it up by alias below.
    spec_set = display.spec.model_dump(mode="json", exclude_none=True, by_alias=True)

    for fname, finfo in DeploymentSpec.model_fields.items():
        yaml_key = finfo.serialization_alias or fname
        out.extend(_doc_lines(_docs(finfo), indent=_INDENT))

        # Special case: ``display_name`` (a.k.a. ``generateName``) is always
        # rendered commented-out. The canonical id is the top-level ``name``;
        # ``generateName`` is a slug seed users opt into. When the model has a
        # value (e.g. ``from_response``), surface it as the example so the
        # user sees what's currently set without making the line authoritative.
        if fname == "display_name":
            value = spec_set.get(yaml_key, _EXAMPLES.get(fname, ""))
            out.append(f"{_INDENT}# {yaml_key}: {_scalar(value)}")
            continue

        if yaml_key in spec_set:
            value = spec_set[yaml_key]
            if fname == "secrets" and isinstance(value, dict):
                out.append(f"{_INDENT}{yaml_key}:")
                for sname, sval in value.items():
                    if sname in secret_comments:
                        out.extend(
                            _doc_lines(
                                secret_comments[sname].split("\n"),
                                indent=_INDENT * 2,
                            )
                        )
                    out.append(f"{_INDENT * 2}{sname}: {_scalar(sval)}")
            else:
                out.append(f"{_INDENT}{yaml_key}: {_scalar(value)}")
                alt = field_alternatives.get(fname)
                if alt is not None:
                    alt_value, alt_note = alt
                    note = f"  # {alt_note}" if alt_note else ""
                    out.append(f"{_INDENT}# {yaml_key}: {alt_value}{note}")
        elif fname in required_set:
            out.append(f"{_INDENT}{_MARKER}{_REQUIRED}")
            out.append(f"{_INDENT}{yaml_key}: ~")
        else:
            example = _EXAMPLES.get(fname, "")
            if isinstance(example, dict):
                out.append(f"{_INDENT}# {yaml_key}:")
                for k, v in example.items():
                    out.append(f"{_INDENT * 2}# {k}: {_scalar(v)}")
            else:
                out.append(f"{_INDENT}# {yaml_key}: {_scalar(example)}")

    return "\n".join(out) + "\n"


def _doc_lines(docs: Iterable[str], *, indent: str) -> list[str]:
    """Return ``## <doc>`` lines at the given indent, trailing-stripped."""
    return [f"{indent}{_MARKER}{d}".rstrip() for d in docs]


def _docs(info: FieldInfo | None) -> tuple[str, ...]:
    """Return the first ``Doc`` marker text on ``info``, split into lines."""
    if info is None:
        return ()
    for marker in info.metadata:
        if isinstance(marker, Doc):
            return tuple(marker.text.split("\n"))
    return ()


def _scalar(value: Any) -> str:
    """Render ``value`` as a YAML scalar suitable for the right-hand side of
    a key line.

    Delegates to PyYAML in mapping context (``{"_": value}``) so its
    emitter applies block-context rules — plain for ``${VAR}``, URLs,
    versions; quoted for reserved words / flow chars / values containing
    ``: `` or `` #``. Two carve-outs: ``None`` emits as ``~`` for parity
    with the required-tilde rendering, and ``""`` emits as ``""`` so the
    push-mode signal isn't hidden as PyYAML's default ``''``.
    """
    if value is None:
        return "~"
    if value == "":
        return '""'
    return yaml.safe_dump(
        {"_": value}, default_flow_style=False, width=1 << 30, allow_unicode=True
    ).rstrip()[3:]
