# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Render a :class:`DeploymentDisplay` as commented apply-shaped YAML.

One ``render()`` walks ``DeploymentDisplay``'s top-level identity fields
(``name``, optionally ``generate_name``) and then ``DeploymentSpec.model_fields``
in declaration order, appending lines to a list. Each spec field renders in
one of three states:

* **set** — ``<key>: <scalar>`` (uncommented), with the field's
  :class:`~llama_agents.cli.display.Doc` text as ``## …`` lines above.
  ``secrets`` is the only nested shape; its child keys render at indent 4.
* **required-but-unset** — ``<key>: ~`` with a ``## Required …`` marker so a
  ``grep`` or eyeball pass finds the gaps that block ``apply``.
* **unset** — ``# <key>: <example>`` (commented out one-liner) with the doc
  above. The schema-fixed ``_EXAMPLES`` table supplies the example value.

Top-level ``name`` follows the set / unset split (no required path — the
server slugifies an id when ``name`` is omitted, so a missing top-level
``name`` is never an apply blocker). ``generate_name`` is special-cased: the
caller opts in via ``scaffold_generate_name`` (only the offline ``deployments
template`` flow does), and when emitted it always renders commented-out under
the identity-tier comment block.

A small ``field_alternatives`` mapping lets the caller surface a
commented-out alternative under a *set* field (used to suggest the detected
git remote under an empty ``repo_url``). Scalar quoting delegates to PyYAML;
``"."`` for ``deployment_file_path`` is force-quoted so the value reads as a
path rather than a YAML float-ish ambiguity.

Mask sentinels (``SECRET_MASK``) inside ``secrets`` and on
``personal_access_token`` are stripped before rendering — the same filter
:func:`~llama_agents.cli.display.strip_masks` applies to ``-o yaml`` /
``-o json`` output, so a ``get -o template | apply`` round-trip can't push a
literal ``********`` back as the value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import yaml
from llama_agents.cli.display import (
    DeploymentDisplay,
    DeploymentSpec,
    Doc,
    strip_masks,
)
from pydantic.fields import FieldInfo

_INDENT = "  "
_MARKER = "## "
_REQUIRED = "Required — set before `apply`."

# Per-field example values shown when a spec field is unset (rendered commented).
_EXAMPLES: dict[str, Any] = {
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
    name_example: str = "my-app",
    scaffold_generate_name: bool = False,
) -> str:
    """Render ``display`` as a commented apply-shaped YAML string.

    Args:
        display: The deployment to render. ``status`` is unconditionally
            omitted; only ``name``, ``generate_name``, and ``spec`` reach the
            output.
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
            it has no required path.
        name_example: Example value rendered for an unset top-level ``name``
            (and ``generate_name`` when it has no model value). Defaults to
            ``"my-app"``; the offline template command passes the cwd name.
        scaffold_generate_name: When ``True``, emit a commented-out
            ``# generate_name: <value>`` line at the identity tier with the
            field's ``Doc`` block above. When ``False`` (the default), the
            field is omitted entirely. Only the offline ``deployments
            template`` flow opts in; ``get -o template`` does not.
    """
    required_set = set(required)
    out: list[str] = []

    for line in head:
        out.append("##" if line == "" else f"{_MARKER}{line}")

    name_docs = _docs(DeploymentDisplay.model_fields["name"])
    out.extend(_doc_lines(name_docs, indent=""))
    if display.name is None:
        out.append(f"# name: {name_example}")
    else:
        out.append(f"name: {_scalar(display.name)}")

    if scaffold_generate_name:
        gn_docs = _docs(DeploymentDisplay.model_fields["generate_name"])
        out.extend(_doc_lines(gn_docs, indent=""))
        gn_value = display.generate_name or name_example
        out.append(f"# generate_name: {_scalar(gn_value)}")

    out.append("spec:")
    spec_dump = display.spec.model_dump(mode="json", exclude_none=True)
    spec_set = strip_masks(spec_dump)

    for fname, finfo in DeploymentSpec.model_fields.items():
        out.extend(_doc_lines(_docs(finfo), indent=_INDENT))

        if fname in spec_set:
            _emit_set_field(
                out, fname, spec_set[fname], secret_comments, field_alternatives
            )
        elif fname in required_set:
            out.append(f"{_INDENT}{_MARKER}{_REQUIRED}")
            out.append(f"{_INDENT}{fname}: ~")
        else:
            _emit_unset_field(out, fname)

    return "\n".join(out) + "\n"


def _emit_set_field(
    out: list[str],
    fname: str,
    value: Any,
    secret_comments: Mapping[str, str],
    field_alternatives: Mapping[str, tuple[str, str]],
) -> None:
    """Append lines for a spec field that has a value."""
    if fname == "secrets" and isinstance(value, dict):
        out.append(f"{_INDENT}{fname}:")
        for sname, sval in value.items():
            if sname in secret_comments:
                out.extend(
                    _doc_lines(secret_comments[sname].split("\n"), indent=_INDENT * 2)
                )
            out.append(f"{_INDENT * 2}{sname}: {_scalar(sval)}")
        return
    out.append(f"{_INDENT}{fname}: {_scalar(value, key=fname)}")
    alt = field_alternatives.get(fname)
    if alt is not None:
        alt_value, alt_note = alt
        note = f"  # {alt_note}" if alt_note else ""
        out.append(f"{_INDENT}# {fname}: {alt_value}{note}")


def _emit_unset_field(out: list[str], fname: str) -> None:
    """Append a commented-out example line for an unset spec field."""
    example = _EXAMPLES.get(fname, "")
    if isinstance(example, dict):
        out.append(f"{_INDENT}# {fname}:")
        for k, v in example.items():
            out.append(f"{_INDENT * 2}# {k}: {_scalar(v)}")
    else:
        out.append(f"{_INDENT}# {fname}: {_scalar(example, key=fname)}")


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


def _scalar(value: Any, *, key: str | None = None) -> str:
    """Render ``value`` as a YAML scalar suitable for the right-hand side of
    a key line.

    Delegates to PyYAML in mapping context (``{"_": value}``) so its
    emitter applies block-context rules — plain for ``${VAR}``, URLs,
    versions; quoted for reserved words / flow chars / values containing
    ``: `` or `` #``. Carve-outs: ``None`` emits as ``~`` for parity with
    the required-tilde rendering, ``""`` emits as ``""`` so the push-mode
    signal isn't hidden as PyYAML's default ``''``, and
    ``deployment_file_path: "."`` is force-quoted so the bare-dot reads as
    a path string rather than YAML's slightly-ambiguous plain scalar.
    """
    if value is None:
        return "~"
    if value == "":
        return '""'
    if key == "deployment_file_path" and value == ".":
        return '"."'
    return yaml.safe_dump(
        {"_": value}, default_flow_style=False, width=1 << 30, allow_unicode=True
    ).rstrip()[3:]
