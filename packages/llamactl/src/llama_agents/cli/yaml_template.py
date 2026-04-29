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

A small ``field_alternatives`` mapping lets the caller surface a
commented-out alternative under a *set* field (used to suggest the detected
git remote under an empty ``repo_url``). Scalar quoting is a hand-rolled
plain-safe check; PyYAML handles the few values that need actual quoting.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import yaml
from llama_agents.cli.display import DeploymentDisplay, DeploymentSpec, Doc
from pydantic.fields import FieldInfo

_INDENT = "  "
_MARKER = "## "
_REQUIRED = "Required — set before `apply`."

# Per-field example values shown when a field is unset (rendered commented).
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
        required: Field names to force-emit as ``<field>: ~`` with a
            ``## Required …`` marker even when unset. Supports the
            top-level ``name`` and any ``DeploymentSpec`` field.
    """
    required_set = set(required)
    out: list[str] = []

    for line in head:
        out.append("##" if line == "" else f"{_MARKER}{line}")

    name_docs = _docs(DeploymentDisplay.model_fields["name"])
    out.extend(_doc_lines(name_docs, indent=""))
    if "name" in required_set:
        out.append(f"{_MARKER}{_REQUIRED}")
        out.append("name: ~")
    else:
        out.append(f"name: {_scalar(display.name)}")

    out.append("spec:")
    spec_set = display.spec.model_dump(mode="json", exclude_none=True)

    for fname, finfo in DeploymentSpec.model_fields.items():
        out.extend(_doc_lines(_docs(finfo), indent=_INDENT))

        if fname in spec_set:
            value = spec_set[fname]
            if fname == "secrets" and isinstance(value, dict):
                out.append(f"{_INDENT}secrets:")
                for sname, sval in value.items():
                    if sname in secret_comments:
                        out.extend(_doc_lines(
                            secret_comments[sname].split("\n"),
                            indent=_INDENT * 2,
                        ))
                    out.append(f"{_INDENT * 2}{sname}: {_scalar(sval)}")
            else:
                out.append(f"{_INDENT}{fname}: {_scalar(value)}")
                alt = field_alternatives.get(fname)
                if alt is not None:
                    alt_value, alt_note = alt
                    note = f"  # {alt_note}" if alt_note else ""
                    out.append(f"{_INDENT}# {fname}: {alt_value}{note}")
        elif fname in required_set:
            out.append(f"{_INDENT}{_MARKER}{_REQUIRED}")
            out.append(f"{_INDENT}{fname}: ~")
        else:
            example = _EXAMPLES.get(fname, "")
            if isinstance(example, dict):
                out.append(f"{_INDENT}# {fname}:")
                for k, v in example.items():
                    out.append(f"{_INDENT * 2}# {k}: {_scalar(v)}")
            else:
                out.append(f"{_INDENT}# {fname}: {_scalar(example)}")

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


# YAML 1.1 reserved words that must not be emitted as plain scalars (they'd
# round-trip as bool / null on parse). Includes the YAML 1.2 trio plus the
# 1.1-only yes/no/on/off (PyYAML's default loader still resolves these).
_PLAIN_RESERVED = frozenset(
    word
    for base in ("true", "false", "null", "yes", "no", "on", "off", "~")
    for word in (base, base.title(), base.upper())
)
# Strict enough to match anything PyYAML's default resolver would coerce to
# int or float.
_NUMERIC_RE = re.compile(
    r"^[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?$"
)


def _plain_safe(value: str) -> bool:
    """Return True if ``value`` is safe as a plain (unquoted) YAML scalar in
    block context.

    Encodes YAML's plain-scalar rules directly rather than asking PyYAML to
    decide — PyYAML's emitter is over-conservative for our use (it quotes
    values containing flow indicators like ``${...}`` even though we never
    emit in flow context). The schema is fixed; the rules are short.
    """
    if not value:
        return False
    if value in _PLAIN_RESERVED:
        return False
    if _NUMERIC_RE.match(value):
        return False
    if value[0] in "!&*[]{,#|>%@`\"'":
        return False
    if value[0] in "?:-" and (len(value) == 1 or value[1] in " \t"):
        return False
    if value[0] in " \t" or value[-1] in " \t:":
        return False
    if "\n" in value or "\t" in value:
        return False
    if ": " in value or " #" in value:
        return False
    return True


def _scalar(value: Any) -> str:
    """Render ``value`` as a YAML scalar suitable for the right-hand side of
    a key line."""
    if value is None:
        return "~"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Push-mode sentinel: emit visibly so the signal isn't hidden.
        if value == "":
            return '""'
        if _plain_safe(value):
            return value
        # Anything not plain-safe goes through PyYAML's quoter.
        text = yaml.safe_dump(
            value, default_flow_style=True, width=1 << 30, allow_unicode=True
        ).rstrip()
        if text.endswith("\n..."):
            text = text[:-4].rstrip()
        elif text.endswith("..."):
            text = text[:-3].rstrip()
        return text
    raise TypeError(
        f"unsupported scalar type for YAML rendering: {type(value).__name__}"
    )
