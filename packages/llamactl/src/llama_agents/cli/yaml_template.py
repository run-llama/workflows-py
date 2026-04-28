# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Render a :class:`DeploymentDisplay` as commented apply-shaped YAML.

The renderer is the chokepoint shared by ``deployments template`` and
``deployments get -o template``. It walks ``DeploymentSpec.model_fields`` in
declaration order, attaches one ``#! `` comment line per field (from the
field's :class:`~llama_agents.cli.display.Doc` marker, optionally overridden
by a per-call ``field_overrides`` mapping), and emits the result via
ruamel.yaml so the comment-bearing structure round-trips on a future load.

ruamel.yaml's stringly-typed comment API is hidden inside :func:`render`;
callers see a typed ``DeploymentDisplay`` going in and a ``str`` coming out.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, Sequence
from typing import Any

from llama_agents.cli.display import DeploymentDisplay, DeploymentSpec, Doc
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

_COMMENT_PREFIX = "#! "
_INDENT = 2
# Rewrite ruamel's ``"<indent># ! "`` artifact to ``"<indent>#! "``.
# Capture the line-start (start-of-string or newline) plus indentation so the
# substitution preserves layout.
_MARKER_RE = re.compile(r"(^|\n)([ \t]*)# ! ")


def _yaml() -> YAML:
    y = YAML()
    y.indent(mapping=_INDENT, sequence=_INDENT * 2, offset=_INDENT)
    y.default_flow_style = False
    return y


def _field_doc(field_name: str) -> str | None:
    info = DeploymentSpec.model_fields.get(field_name)
    if info is None:
        return None
    for marker in info.metadata:
        if isinstance(marker, Doc):
            return marker.text
    return None


def _attach_before_comment(cm: CommentedMap, key: str, text: str, indent: int) -> None:
    """Attach a ``#! <text>`` line above ``key`` in a CommentedMap.

    ruamel's ``yaml_set_comment_before_after_key`` always prefixes the body
    with ``"# "`` (with a trailing space). We pass ``"!<text>"`` so the
    rendered line is ``"# !<text>"``; the post-pass in :func:`_finalize`
    collapses ``"# !"`` → ``"#!"`` for the documented prefix.
    """
    cm.yaml_set_comment_before_after_key(
        key, before=f"! {text}", indent=indent
    )


def _finalize(text: str) -> str:
    """Collapse ruamel's ``# !`` artifact back to the ``#!`` prefix.

    ruamel renders comments as ``"<indent># " + body``. We pass
    ``"! <body>"`` so each line surfaces as ``"<indent># ! <body>"``; this
    pass rewrites every such line to ``"<indent>#! <body>"`` so the output
    matches the documented ``#!`` prefix.
    """
    return _MARKER_RE.sub(r"\1\2#! ", text)


def _coerce_scalar(value: Any) -> Any:
    """Force empty strings to render as ``""`` (double-quoted) rather than
    ruamel's bare-empty-scalar default. The empty repo_url is a meaningful
    signal (push-mode); we want it visibly present in the dump."""
    if isinstance(value, str) and value == "":
        return DoubleQuotedScalarString("")
    return value


def render(
    display: DeploymentDisplay,
    *,
    head: Sequence[str] = (),
    field_overrides: Mapping[str, str] = {},
    secret_comments: Mapping[str, str] = {},
    scaffold_unset: bool = False,
) -> str:
    """Render ``display`` as a commented apply-shaped YAML string.

    Args:
        display: The deployment to render. ``status`` is unconditionally
            omitted; only ``name`` and ``spec`` reach the output.
        head: Lines emitted at the top of the output as ``#! `` comments,
            before ``name:``. Empty-string entries become a blank ``#!`` line.
        field_overrides: Map of ``DeploymentSpec`` field name → comment text.
            Overrides the ``Doc`` marker for that one render. Unknown keys
            raise ``ValueError``.
        secret_comments: Map of secret name → comment text. Each becomes a
            ``#! `` line above the matching key inside the ``secrets:`` block.
        scaffold_unset: When ``True``, append a tail block listing each
            spec field that has a ``Doc`` and is *not* set on the input as a
            commented-out hint. Used by ``deployments template`` so the user
            can uncomment to add optional fields.
    """

    unknown = [k for k in field_overrides if k not in DeploymentSpec.model_fields]
    if unknown:
        raise ValueError(
            f"field_overrides keys not on DeploymentSpec: {sorted(unknown)}"
        )

    spec_set = display.spec.model_dump(mode="json", exclude_none=True)

    spec_map = CommentedMap()
    # Walk in declaration order — never sort, never let YAML re-order.
    last_set_key: str | None = None
    for field_name in DeploymentSpec.model_fields.keys():
        if field_name not in spec_set:
            continue
        value = spec_set[field_name]
        if field_name == "secrets" and isinstance(value, dict):
            secrets_map = CommentedMap()
            for sname, sval in value.items():
                secrets_map[sname] = _coerce_scalar(sval)
                comment = secret_comments.get(sname)
                if comment:
                    _attach_before_comment(
                        secrets_map, sname, comment, indent=_INDENT * 2
                    )
            spec_map[field_name] = secrets_map
        else:
            spec_map[field_name] = _coerce_scalar(value)

        comment_text = field_overrides.get(field_name) or _field_doc(field_name)
        if comment_text:
            _attach_before_comment(
                spec_map, field_name, comment_text, indent=_INDENT
            )
        last_set_key = field_name

    root = CommentedMap()
    root["name"] = display.name
    name_doc = field_overrides.get("name")
    if name_doc is None:
        # ``name`` doc lives on DeploymentDisplay, not DeploymentSpec — read it
        # off the parent model so the renderer stays a single chokepoint.
        for marker in DeploymentDisplay.model_fields["name"].metadata:
            if isinstance(marker, Doc):
                name_doc = marker.text
                break
    if name_doc:
        _attach_before_comment(root, "name", name_doc, indent=0)
    root["spec"] = spec_map

    yaml = _yaml()
    buf = io.StringIO()
    if head:
        for line in head:
            if line == "":
                buf.write("#!\n")
            else:
                buf.write(f"{_COMMENT_PREFIX}{line}\n")
    yaml.dump(root, buf)
    out = _finalize(buf.getvalue())

    if scaffold_unset:
        out += _scaffold_tail(spec_set)

    # Quiet ruff: ``last_set_key`` is informational; future overrides may use it.
    _ = last_set_key
    return out


def _scaffold_tail(spec_set: Mapping[str, Any]) -> str:
    """Produce a trailing block of commented-out hints for unset spec fields.

    Each entry is two lines: a ``#! `` instruction (the field's Doc) and a
    commented-out ``# field: <example>`` line under the spec block. Indented
    to fall inside ``spec:`` so uncommenting yields a structurally-correct
    YAML key.
    """
    lines: list[str] = []
    for field_name, info in DeploymentSpec.model_fields.items():
        if field_name in spec_set:
            continue
        doc = next(
            (m.text for m in info.metadata if isinstance(m, Doc)),
            None,
        )
        if doc is None:
            continue
        if not lines:
            lines.append("")
            lines.append("#! Optional fields (uncomment to use):")
        indent_outer = " " * _INDENT
        indent_inner = " " * (_INDENT * 2)
        lines.append(f"{indent_outer}{_COMMENT_PREFIX}{doc}")
        example = _scaffold_example(field_name)
        if isinstance(example, str):
            lines.append(f"{indent_outer}# {field_name}: {example}")
        else:
            # Multi-line scaffold (e.g. secrets dict): emit per-line.
            lines.append(f"{indent_outer}# {field_name}:")
            for sub in example:
                lines.append(f"{indent_inner}# {sub}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _scaffold_example(field_name: str) -> str | list[str]:
    """Pick a sensible commented-out example for a given spec field.

    Returns either a scalar (rendered inline as ``# field: <value>``) or a
    list of body lines (rendered as ``# field:`` followed by ``# <body>``
    lines indented one extra level).
    """
    if field_name == "secrets":
        return ["MY_SECRET: ${MY_SECRET}"]
    if field_name == "suspended":
        return "false"
    if field_name == "appserver_version":
        return '"0.5.0"'
    if field_name == "personal_access_token":
        return "${GITHUB_TOKEN}"
    if field_name == "repo_url":
        return '"https://github.com/owner/repo"'
    if field_name == "git_ref":
        return "main"
    if field_name == "deployment_file_path":
        return '"."'
    if field_name == "display_name":
        return '"My App"'
    return '""'


def load_commented(text: str) -> CommentedMap:
    """Load ``text`` into a comment-preserving ruamel ``CommentedMap``.

    Made available for the future ``--annotate-on-error`` slice; callers
    today don't need to touch ruamel directly.
    """
    yaml = _yaml()
    loaded = yaml.load(text)
    if not isinstance(loaded, CommentedMap):
        raise ValueError("YAML document does not parse to a mapping")
    return loaded
