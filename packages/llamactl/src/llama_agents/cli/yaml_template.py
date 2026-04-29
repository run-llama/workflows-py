# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Render a :class:`DeploymentDisplay` as commented apply-shaped YAML.

The renderer is the chokepoint shared by ``deployments template`` and
``deployments get -o template``. It walks ``DeploymentSpec.model_fields`` in
declaration order, building one :class:`FieldBlock` per output block, then
emits text by walking blocks in order.

Each block carries its state (``set`` / ``required`` / ``unset``), its docs
(rendered as ``## ...`` lines above the key), an optional alternative line
(rendered as a commented-out sibling under set fields), and any nested
children (the ``secrets`` dict — the schema's only nested case). The fixed
schema and one-deep nesting mean we hand-roll the emit instead of leaning on
a general YAML library; PyYAML is used only for scalar quoting decisions.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml
from llama_agents.cli.display import DeploymentDisplay, DeploymentSpec, Doc
from pydantic.fields import FieldInfo

_INDENT = 2
_MARKER = "## "
_REQUIRED_LINE = "Required — set before `apply`."

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


FieldState = Literal["set", "required", "unset"]


@dataclass(frozen=True)
class FieldBlock:
    """One output block: optional docs, the key line, optional alternative.

    A block always renders into a contiguous run of lines. ``state``
    determines the key-line shape:

    * ``set`` — ``<key>: <scalar>`` (uncommented).
    * ``required`` — ``<key>: ~`` with a ``## Required …`` line in the docs.
    * ``unset`` — ``# <key>: <example>`` (commented out, with example value).

    ``children`` carries nested blocks (the ``secrets`` dict). When the
    parent is ``unset``, ``# `` is propagated to each child line so the
    whole block reads as commented-out from a single sibling-key column.
    """

    key: str
    indent: int
    state: FieldState
    value: Any = None
    docs: tuple[str, ...] = ()
    alternative: tuple[str, str] | None = None
    children: tuple[FieldBlock, ...] = field(default_factory=tuple)

    def lines(self) -> Iterator[str]:
        pad = " " * self.indent
        commented = self.state == "unset"
        # Docs render above the key at the field's own indent. A doc above a
        # commented (``unset``) field still uses ``## `` (not ``# ## ``):
        # only the key line and its children carry the leading ``# ``.
        for doc_line in self.docs:
            yield f"{pad}{_MARKER}{doc_line}".rstrip()
        if self.state == "required":
            yield f"{pad}{_MARKER}{_REQUIRED_LINE}"
            yield f"{pad}{self.key}: ~"
            return
        prefix = f"{pad}# " if commented else pad
        if self.children:
            yield f"{prefix}{self.key}:"
            for child in self.children:
                for child_line in child.lines():
                    if commented:
                        c_pad_n = len(child_line) - len(child_line.lstrip(" "))
                        yield child_line[:c_pad_n] + "# " + child_line[c_pad_n:]
                    else:
                        yield child_line
        else:
            yield f"{prefix}{self.key}: {_scalar(self.value)}"
        if self.state == "set" and self.alternative is not None:
            alt_value, alt_note = self.alternative
            note = f"  # {alt_note}" if alt_note else ""
            yield f"{pad}# {self.key}: {alt_value}{note}"


def _scalar(value: Any) -> str:
    """Render ``value`` as a YAML scalar suitable for the right-hand side of
    a key line. Prefers unquoted plain scalars where round-trip is safe;
    falls back to PyYAML's quoting decision otherwise.
    """
    if value is None:
        return "~"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Push-mode sentinel: emit visibly so the signal isn't hidden as a
        # bare empty scalar.
        if value == "":
            return '""'
        if _is_safe_plain(value):
            return value
        return _quote(value)
    raise TypeError(
        f"unsupported scalar type for YAML rendering: {type(value).__name__}"
    )


def _is_safe_plain(value: str) -> bool:
    """Return True if ``value`` round-trips losslessly as a plain (unquoted)
    block-context YAML scalar.

    Conservative on YAML indicators that change parser behavior (``: ``,
    `` #``, leading flow / comment / anchor chars), but permissive on
    interior characters that ruamel handled silently (``${...}``, URLs,
    version strings, ``.``).
    """
    if not value:
        return False
    if value[0] in "!&*[]{},#|>%@`\"'":
        return False
    if value[0] in "?:-" and (len(value) == 1 or value[1] in " \t"):
        return False
    if value.endswith(":"):
        return False
    if value != value.strip() or "\n" in value or "\t" in value:
        return False
    if ": " in value or " #" in value:
        return False
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return False
    return parsed == value


def _quote(value: str) -> str:
    """Return ``value`` quoted via PyYAML, stripped of document framing."""
    text = yaml.safe_dump(
        value, default_flow_style=True, width=1 << 30, allow_unicode=True
    ).rstrip()
    if text.endswith("\n..."):
        text = text[:-4].rstrip()
    elif text.endswith("..."):
        text = text[:-3].rstrip()
    return text


def _docs_for(info: FieldInfo | None) -> tuple[str, ...]:
    """Return the first ``Doc`` marker text on ``info``, split into lines."""
    if info is None:
        return ()
    for marker in info.metadata:
        if isinstance(marker, Doc):
            return tuple(marker.text.split("\n"))
    return ()


def _build_blocks(
    display: DeploymentDisplay,
    *,
    secret_comments: Mapping[str, str],
    field_alternatives: Mapping[str, tuple[str, str]],
    required: set[str],
) -> tuple[FieldBlock, list[FieldBlock]]:
    """Build the top-level ``name`` block and the spec-field blocks."""

    name_state: FieldState = "required" if "name" in required else "set"
    name_block = FieldBlock(
        key="name",
        indent=0,
        state=name_state,
        value=display.name if name_state == "set" else None,
        docs=_docs_for(DeploymentDisplay.model_fields.get("name")),
    )

    # ``exclude_none=True`` defines "set" for our purposes — wire types come
    # in with explicit ``None`` for unset fields, not absent keys.
    spec_set = display.spec.model_dump(mode="json", exclude_none=True)

    spec_blocks: list[FieldBlock] = []
    for fname, finfo in DeploymentSpec.model_fields.items():
        docs = _docs_for(finfo)
        if fname in spec_set:
            value = spec_set[fname]
            if fname == "secrets" and isinstance(value, dict):
                children = tuple(
                    FieldBlock(
                        key=sname,
                        indent=_INDENT * 2,
                        state="set",
                        value=sval,
                        docs=tuple(secret_comments[sname].split("\n"))
                        if sname in secret_comments
                        else (),
                    )
                    for sname, sval in value.items()
                )
                spec_blocks.append(
                    FieldBlock(
                        key=fname,
                        indent=_INDENT,
                        state="set",
                        docs=docs,
                        children=children,
                    )
                )
            else:
                spec_blocks.append(
                    FieldBlock(
                        key=fname,
                        indent=_INDENT,
                        state="set",
                        value=value,
                        docs=docs,
                        alternative=field_alternatives.get(fname),
                    )
                )
        elif fname in required:
            spec_blocks.append(
                FieldBlock(
                    key=fname,
                    indent=_INDENT,
                    state="required",
                    docs=docs,
                )
            )
        else:
            example = _EXAMPLES.get(fname, "")
            if isinstance(example, dict):
                children = tuple(
                    FieldBlock(
                        key=sname,
                        indent=_INDENT * 2,
                        state="set",
                        value=sval,
                    )
                    for sname, sval in example.items()
                )
                spec_blocks.append(
                    FieldBlock(
                        key=fname,
                        indent=_INDENT,
                        state="unset",
                        docs=docs,
                        children=children,
                    )
                )
            else:
                spec_blocks.append(
                    FieldBlock(
                        key=fname,
                        indent=_INDENT,
                        state="unset",
                        value=example,
                        docs=docs,
                    )
                )
    return name_block, spec_blocks


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
            block. Ignored for unset secrets.
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
    name_block, spec_blocks = _build_blocks(
        display,
        secret_comments=secret_comments,
        field_alternatives=field_alternatives,
        required=required_set,
    )

    out: list[str] = []
    for line in head:
        out.append("##" if line == "" else f"{_MARKER}{line}")

    out.extend(name_block.lines())
    out.append("spec:")
    for block in spec_blocks:
        out.extend(block.lines())

    return "\n".join(out) + "\n"
