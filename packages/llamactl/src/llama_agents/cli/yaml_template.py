# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Render a :class:`DeploymentDisplay` as commented apply-shaped YAML.

The renderer is the chokepoint shared by ``deployments template`` and
``deployments get -o template``. It walks ``DeploymentSpec.model_fields`` in
declaration order, emitting one block per field in one of three states:

* **set** — the value is present on the input; rendered uncommented via
  ruamel.yaml with the field's :class:`~llama_agents.cli.display.Doc` text
  (or a per-call ``field_overrides`` value) attached as ``#! `` lines above.
* **required-but-unset** — the field name is in the ``required`` arg; rendered
  as ``<field>: ~`` (uncommented) with a trailing ``#! Required - set before
  apply.`` line so a ``grep`` for ``Required`` or ``~`` finds the gaps.
* **unset** — neither set nor required; rendered as a commented-out
  ``# <field>: <example>`` one-liner with the Doc above it.

A small ``field_alternatives`` mapping lets the caller surface a commented-out
alternative value directly under a *set* field (used by the in-git-repo case
to show the auto-detected ``git remote get-url origin`` as a one-keystroke
swap-in for an empty ``repo_url``).

ruamel.yaml's stringly-typed comment API is hidden inside :func:`render`;
callers see a typed ``DeploymentDisplay`` going in and a ``str`` coming out.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from llama_agents.cli.display import DeploymentDisplay, DeploymentSpec, Doc
from pydantic.fields import FieldInfo
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

# Sentinel value of ``DeploymentSpec.repo_url`` indicating push-mode (the CLI
# pushes the local working tree on apply rather than pointing at a remote).
PUSH_MODE_REPO_URL = ""

_COMMENT_PREFIX = "#! "
_INDENT = 2
_REQUIRED_LINE = "Required — set before `apply`."
# Rewrite ruamel's ``"<indent># ! "`` artifact to ``"<indent>#! "``.
# Capture the line-start (start-of-string or newline) plus indentation so the
# substitution preserves layout.
_MARKER_RE = re.compile(r"(^|\n)([ \t]*)# ! ")
# Match a top-level (indent=0) or spec-level (indent=2) bare-key line so we
# can rewrite ``foo:\n`` (ruamel's null emission) to ``foo: ~\n`` for
# required-unset fields.
_BARE_KEY_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s*$")
# Inline form (no anchors / trailing slack) for matching a key on a single
# line that has already been stripped of its indent.
_INLINE_KEY_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):")


# Commented-out example values per field. Used when the field is in the
# ``unset`` state (not set on the input and not in ``required``). Quoting on
# string examples matches how ruamel emits the same value in the set state,
# so the user uncommenting the line gets an apply-shaped result.
_EXAMPLES: dict[str, Any] = {
    "display_name": "My App",
    # Example shows a real URL; the push-mode sentinel is documented separately
    # via :data:`PUSH_MODE_REPO_URL`.
    "repo_url": DoubleQuotedScalarString("https://github.com/owner/repo"),
    "deployment_file_path": DoubleQuotedScalarString("."),
    "git_ref": "main",
    "appserver_version": DoubleQuotedScalarString("0.5.0"),
    "suspended": False,
    "secrets": {"MY_SECRET": "${MY_SECRET}"},
    "personal_access_token": "${GITHUB_TOKEN}",
}


def _yaml() -> YAML:
    y = YAML()
    y.indent(mapping=_INDENT, sequence=_INDENT * 2, offset=_INDENT)
    y.default_flow_style = False
    return y


def _extract_doc(info: FieldInfo | None) -> str | None:
    """Return the text of the first ``Doc`` marker on ``info``, if any."""
    if info is None:
        return None
    for marker in info.metadata:
        if isinstance(marker, Doc):
            return marker.text
    return None


def _spec_field_doc(field_name: str) -> str | None:
    return _extract_doc(DeploymentSpec.model_fields.get(field_name))


def _name_doc() -> str | None:
    return _extract_doc(DeploymentDisplay.model_fields.get("name"))


def _with_required(doc_text: str) -> str:
    """Append ``_REQUIRED_LINE`` to ``doc_text``, separated by a newline."""
    return (doc_text + "\n" if doc_text else "") + _REQUIRED_LINE


def _attach_before_comment(cm: CommentedMap, key: str, text: str, indent: int) -> None:
    """Attach ``text`` as ``#! `` lines above ``key`` in a CommentedMap.

    ``text`` may contain ``\\n`` — each line is emitted as its own comment.
    ruamel's ``yaml_set_comment_before_after_key`` always prefixes the body
    with ``"# "`` (with a trailing space). We pass ``"!<line>"`` per line so
    the rendered output is ``"# !<line>"``; the post-pass in :func:`_finalize`
    collapses ``"# !"`` → ``"#!"`` for the documented prefix.
    """
    body = "\n".join(f"! {line}" for line in text.split("\n"))
    cm.yaml_set_comment_before_after_key(key, before=body, indent=indent)


def _finalize(text: str) -> str:
    """Collapse ruamel's ``# !`` artifact back to the ``#!`` prefix.

    ruamel renders comments as ``"<indent># " + body``. We pass
    ``"! <body>"`` so each line surfaces as ``"<indent># ! <body>"``; this
    pass rewrites every such line to ``"<indent>#! <body>"`` so the output
    matches the documented ``#!`` prefix.
    """
    return _MARKER_RE.sub(r"\1\2#! ", text)


def _coerce_scalar(field_name: str, value: Any) -> Any:
    """Force string representations that ruamel would otherwise emit ambiguously.

    * Empty strings render as ``""`` (double-quoted) rather than ruamel's
      bare-empty-scalar default — push-mode is a meaningful signal we want
      visibly present in the dump.
    * ``deployment_file_path`` always renders quoted (so ``"."`` doesn't read
      as the YAML "current document" sentinel and matches the offline scaffold
      shape).
    """
    if isinstance(value, str):
        if value == "":
            return DoubleQuotedScalarString("")
        if field_name == "deployment_file_path":
            return DoubleQuotedScalarString(value)
    return value


def render(
    display: DeploymentDisplay,
    *,
    head: Sequence[str] = (),
    field_overrides: Mapping[str, str] = {},
    secret_comments: Mapping[str, str] = {},
    field_alternatives: Mapping[str, tuple[str, str]] = {},
    required: Iterable[str] = (),
) -> str:
    """Render ``display`` as a commented apply-shaped YAML string.

    Args:
        display: The deployment to render. ``status`` is unconditionally
            omitted; only ``name`` and ``spec`` reach the output.
        head: Lines emitted at the top of the output as ``#! `` comments,
            before ``name:``. Empty-string entries become a blank ``#!`` line.
        field_overrides: Map of field name (on ``DeploymentSpec`` or the
            top-level ``name``) → comment text. Overrides the ``Doc`` marker
            for that one render. Unknown keys raise ``ValueError``.
        secret_comments: Map of secret name → comment text. Each becomes a
            ``#! `` line above the matching key inside the ``secrets:`` block.
        field_alternatives: Map of field name → ``(suggestion, annotation)``.
            For each *set* field present in the map, a commented-out
            ``# <field>: <suggestion>  # <annotation>`` line is emitted
            directly under the field. The mapping is silently ignored for
            fields in the ``required`` or ``unset`` states.
        required: Field names to force-emit as ``<field>: ~`` with a trailing
            ``Required`` marker even when unset. Supports both
            ``DeploymentSpec`` field names and the top-level ``name``.
    """

    valid_keys = set(DeploymentSpec.model_fields) | {"name"}
    unknown = [k for k in field_overrides if k not in valid_keys]
    if unknown:
        raise ValueError(
            f"field_overrides keys not on DeploymentSpec or DeploymentDisplay: {sorted(unknown)}"
        )

    spec_set = display.spec.model_dump(mode="json", exclude_none=True)
    required_set = set(required)
    commented_fields: set[str] = set()
    required_spec_fields: set[str] = set()
    for field_name in DeploymentSpec.model_fields:
        if field_name in spec_set:
            continue
        if field_name in required_set:
            required_spec_fields.add(field_name)
        else:
            commented_fields.add(field_name)

    spec_map = CommentedMap()
    for field_name in DeploymentSpec.model_fields.keys():
        if field_name in spec_set:
            value = spec_set[field_name]
            if field_name == "secrets" and isinstance(value, dict):
                secrets_map = CommentedMap()
                for sname, sval in value.items():
                    secrets_map[sname] = _coerce_scalar(sname, sval)
                    comment = secret_comments.get(sname)
                    if comment:
                        _attach_before_comment(
                            secrets_map, sname, comment, indent=_INDENT * 2
                        )
                spec_map[field_name] = secrets_map
            else:
                spec_map[field_name] = _coerce_scalar(field_name, value)
        elif field_name in required_spec_fields:
            spec_map[field_name] = None
        else:
            example = _EXAMPLES.get(field_name, "")
            if isinstance(example, dict):
                inner = CommentedMap()
                for k, v in example.items():
                    inner[k] = v
                spec_map[field_name] = inner
            else:
                spec_map[field_name] = example

        doc_text = field_overrides.get(field_name) or _spec_field_doc(field_name) or ""
        if field_name in required_spec_fields:
            doc_text = _with_required(doc_text)
        if doc_text:
            _attach_before_comment(spec_map, field_name, doc_text, indent=_INDENT)

    name_is_required = "name" in required_set
    root = CommentedMap()
    root["name"] = None if name_is_required else display.name
    name_doc = field_overrides.get("name") or _name_doc() or ""
    if name_is_required:
        name_doc = _with_required(name_doc)
    if name_doc:
        _attach_before_comment(root, "name", name_doc, indent=0)
    root["spec"] = spec_map

    yaml = _yaml()
    buf = io.StringIO()
    yaml.dump(root, buf)
    out = _finalize(buf.getvalue())

    out = _comment_out_unset_fields(out, commented_fields)
    out = _convert_required_to_tilde(
        out,
        required_spec_fields=required_spec_fields,
        name_is_required=name_is_required,
    )
    out = _inject_alternatives(out, field_alternatives, spec_set)

    if head:
        head_lines = [
            "#!" if line == "" else f"{_COMMENT_PREFIX}{line}" for line in head
        ]
        out = "\n".join(head_lines) + "\n" + out

    return out


def _comment_out_unset_fields(text: str, commented_fields: set[str]) -> str:
    """Prefix ``# `` to the field line (and any nested children) for each
    spec field in ``commented_fields``.

    Walks line by line. When a top-level spec key (indent=2) matches one of
    the commented fields, the line gets ``# `` after its indent. Deeper-indent
    lines that follow get the same treatment until a sibling key (indent=2)
    reasserts the spec block, or the spec block ends.
    """
    if not commented_fields:
        return text
    lines = text.splitlines()
    new_lines: list[str] = []
    in_spec = False
    in_commented_block = False
    spec_field_indent = _INDENT
    for line in lines:
        stripped = line.lstrip(" ")
        indent_n = len(line) - len(stripped)
        if line.startswith("spec:"):
            in_spec = True
            in_commented_block = False
            new_lines.append(line)
            continue
        if not in_spec:
            new_lines.append(line)
            continue
        if not stripped:
            # Blank line — passes through; resets nothing.
            new_lines.append(line)
            continue
        # Comment lines (the Docs above each field) keep state. A `#! ` line at
        # the spec_field_indent column belongs to the *next* field; don't comment.
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        if indent_n == spec_field_indent:
            key_match = _INLINE_KEY_RE.match(stripped)
            if key_match and key_match.group(1) in commented_fields:
                in_commented_block = True
                new_lines.append(line[:indent_n] + "# " + line[indent_n:])
            else:
                in_commented_block = False
                new_lines.append(line)
        elif indent_n > spec_field_indent and in_commented_block:
            # Nested under a commented field — comment out at the spec indent
            # column so the inner ``# MY_SECRET: …`` lines line up under
            # ``# secrets:``.
            new_lines.append(line[:indent_n] + "# " + line[indent_n:])
        else:
            in_commented_block = False
            new_lines.append(line)
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + suffix


def _convert_required_to_tilde(
    text: str,
    *,
    required_spec_fields: set[str],
    name_is_required: bool,
) -> str:
    """Rewrite bare-key ``foo:\\n`` lines to ``foo: ~\\n`` for required fields.

    ruamel emits ``None`` as a bare key with no value (``foo:`` followed by a
    newline). We want a visible ``~`` so a quick scan or ``grep`` finds the
    gaps that block ``apply``.
    """
    if not required_spec_fields and not name_is_required:
        return text
    new_lines: list[str] = []
    for line in text.splitlines():
        m = _BARE_KEY_RE.match(line)
        if m:
            indent_n = len(m.group("indent"))
            key = m.group("key")
            is_top_required = indent_n == 0 and name_is_required and key == "name"
            is_spec_required = indent_n == _INDENT and key in required_spec_fields
            if is_top_required or is_spec_required:
                new_lines.append(f"{m.group('indent')}{key}: ~")
                continue
        new_lines.append(line)
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + suffix


def _inject_alternatives(
    text: str,
    alternatives: Mapping[str, tuple[str, str]],
    spec_set: Mapping[str, Any],
) -> str:
    """Insert a commented-out ``# <field>: <suggestion>  # <annotation>`` line
    directly under each set field listed in ``alternatives``.

    The mapping is silently ignored for fields not in ``spec_set`` — the
    alternative is conceptually "another value you could use *instead*" of
    the value that's there, not "an example".
    """
    if not alternatives:
        return text
    applicable = {k: v for k, v in alternatives.items() if k in spec_set}
    if not applicable:
        return text
    new_lines: list[str] = []
    for line in text.splitlines():
        new_lines.append(line)
        stripped = line.lstrip(" ")
        if stripped.startswith("#"):
            continue
        for field_name, (suggestion, annotation) in applicable.items():
            prefix = f"{field_name}:"
            if not stripped.startswith(prefix):
                continue
            # Make sure it's the actual key, not a longer key that starts the
            # same way (e.g. ``personal_access_token:`` vs ``personal:``).
            after = stripped[len(prefix) :]
            if after and not (after.startswith(" ") or after.startswith("\t")):
                continue
            indent_n = len(line) - len(stripped)
            indent = " " * indent_n
            note = f"  # {annotation}" if annotation else ""
            new_lines.append(f"{indent}# {field_name}: {suggestion}{note}")
            break
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(new_lines) + suffix


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
