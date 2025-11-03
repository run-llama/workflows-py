# SPDX-License-Identifier: MIT
"""
Generate Markdown reference docs for the Workflows Server API from an OpenAPI JSON.

Usage:
  uv run python scripts/generate_server_reference_docs.py \
    --openapi openapi.json \
    --out docs/src/content/docs/workflows-api-reference

If --openapi is not provided or the file is missing, this script attempts
to import and call WorkflowServer().openapi_schema() to obtain the schema
without spinning up a server.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union


Json = Union[dict, list, str, int, float, bool, None]


@dataclass
class Config:
    openapi_path: Optional[Path]
    out_dir: Path


def load_openapi(config: Config) -> Dict[str, Any]:
    if config.openapi_path is not None and config.openapi_path.exists():
        with config.openapi_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback: generate via in-process import
    try:
        from workflows.server.server import WorkflowServer  # type: ignore

        schema = WorkflowServer().openapi_schema()
        return schema
    except Exception as e:  # pragma: no cover - fallback only
        raise RuntimeError(
            "Failed to load OpenAPI. Provide --openapi pointing to an existing JSON, "
            f"or ensure server optional deps are installed. Error: {e}"
        ) from e


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def md_escape(text: str) -> str:
    # Minimal escaping for markdown tables and headings
    return (
        text.replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "\u0060")
    )


def _to_jsonable(value: object) -> Json:
    """Best-effort conversion of arbitrary mappings/sequences to JSON-serializable types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value  # type: ignore[return-value]
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    # Fallback to string representation to avoid serialization errors
    return str(value)


def render_schema(schema: object) -> str:
    """Legacy JSON renderer (kept as fallback)."""
    try:
        jsonable = _to_jsonable(schema)
        return "```json\n" + json.dumps(jsonable, indent=2, ensure_ascii=False) + "\n```\n"
    except Exception:
        return "```json\n{}\n```\n"


def resolve_ref(ref: str, components: Mapping[str, Any]) -> Tuple[str, Optional[object]]:
    # Specs like '#/components/schemas/Handler'
    if not ref.startswith("#/"):
        return ref, None
    parts = ref.split("/")
    if len(parts) < 4:
        return ref, None
    _, components_key, group, name = parts[0], parts[1], parts[2], "/".join(parts[3:])
    if components_key != "components":
        return ref, None
    group_map = components.get(group, {}) if isinstance(components, Mapping) else {}
    node = group_map.get(name)
    return name, node


def _linkify_anchor(text: str) -> str:
    """Create a local markdown link anchor for a header name.

    Note: Many doc systems generate lowercase-hyphenated anchors from headers.
    We follow that convention for intra-page links.
    """
    anchor = text.strip().lower().replace(" ", "-")
    return f"[`{text}`](#{anchor})"


def _schema_type_repr(schema: Mapping[str, Any], components: Mapping[str, Any]) -> str:
    """Return a concise, human-friendly type representation for a schema."""
    # $ref
    if "$ref" in schema:
        ref_name, _ = resolve_ref(str(schema["$ref"]), components)
        return ref_name

    # oneOf/anyOf/allOf
    for key in ("oneOf", "anyOf", "allOf"):
        if key in schema and isinstance(schema[key], list):
            variants = [v for v in schema[key] if isinstance(v, Mapping)]
            inner = ", ".join(_schema_type_repr(v, components) for v in variants)
            label = {
                "oneOf": "one of",
                "anyOf": "any of",
                "allOf": "all of",
            }[key]
            return f"{label} ({inner})"

    t = str(schema.get("type", "")).strip()
    fmt = str(schema.get("format", "")).strip()

    if t == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            return f"array of {_schema_type_repr(items, components)}"
        return "array"

    if t == "object" or (not t and ("properties" in schema or "additionalProperties" in schema)):
        # Map/dictionary form
        if "additionalProperties" in schema and not schema.get("properties"):
            ap = schema.get("additionalProperties")
            if isinstance(ap, Mapping):
                return f"map[string -> {_schema_type_repr(ap, components)}]"
            return "map[string -> any]"
        return "object"

    if t:
        return f"{t}{f' ({fmt})' if fmt else ''}"

    # Enum-only schema without explicit type
    if "enum" in schema:
        try:
            enum_vals = ", ".join(map(str, schema.get("enum", [])))
            return f"enum ({enum_vals})"
        except Exception:
            return "enum"

    return "unknown"


def _render_enum(schema: Mapping[str, Any]) -> str:
    if not isinstance(schema.get("enum"), list):
        return ""
    try:
        values = ", ".join(md_escape(str(v)) for v in schema["enum"])
    except Exception:
        values = ""
    return f"- **Allowed values**: {values}\n"


def _render_examples(schema: Mapping[str, Any]) -> str:
    example = schema.get("example")
    if example is None:
        return ""
    try:
        return "**Example**\n\n" + render_schema(example)
    except Exception:
        return ""


def _render_object_properties_table(schema: Mapping[str, Any], components: Mapping[str, Any]) -> str:
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping) or not properties:
        return ""
    required_set = set(schema.get("required", []) if isinstance(schema.get("required"), list) else [])
    rows: List[str] = [
        "| Field | Type | Required | Description | Default |",
        "| --- | --- | :---: | --- | --- |",
    ]
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, Mapping):
            continue
        typ = _schema_type_repr(prop_schema, components)
        desc = str(prop_schema.get("description", "")).strip()
        default = prop_schema.get("default")
        default_str = md_escape(json.dumps(default)) if default is not None else ""
        rows.append(
            f"| `{md_escape(str(prop_name))}` | {md_escape(typ)} | {'yes' if prop_name in required_set else 'no'} | {md_escape(desc)} | {default_str} |"
        )
    return "\n".join(rows) + "\n\n"


def render_schema_readable(schema: object, components: Mapping[str, Any]) -> str:
    """Render a schema in a human-friendly format.

    - Objects: table of properties
    - Arrays/Primitives: bullet list of key details
    - oneOf/anyOf/allOf: enumerated summary
    Falls back to JSON if structure is unknown.
    """
    if not isinstance(schema, Mapping):
        return render_schema(schema)

    # Follow $ref once for readability
    if "$ref" in schema:
        ref_name, ref_schema = resolve_ref(str(schema["$ref"]), components)
        heading = f"- Schema: {_linkify_anchor(ref_name)}\n\n"
        if isinstance(ref_schema, Mapping):
            return heading + render_schema_readable(ref_schema, components)
        return heading

    # oneOf/anyOf/allOf
    for key in ("oneOf", "anyOf", "allOf"):
        if key in schema and isinstance(schema[key], list) and schema[key]:
            label = {"oneOf": "One of", "anyOf": "Any of", "allOf": "All of"}[key]
            lines: List[str] = [f"**{label}**\n"]
            for variant in schema[key]:
                if not isinstance(variant, Mapping):
                    continue
                lines.append(f"- {_schema_type_repr(variant, components)}\n")
            # If this union also describes properties, show them
            props_tbl = _render_object_properties_table(schema, components)
            if props_tbl:
                lines.append("\n" + props_tbl)
            lines.append(_render_enum(schema))
            lines.append(_render_examples(schema))
            return "".join(lines)

    # Objects
    is_object = schema.get("type") == "object" or "properties" in schema or "additionalProperties" in schema
    if is_object:
        lines: List[str] = []
        # If map-like
        if "additionalProperties" in schema and not schema.get("properties"):
            ap = schema.get("additionalProperties")
            if isinstance(ap, Mapping):
                lines.append(f"- Type: map[string -> {_schema_type_repr(ap, components)}]\n")
            else:
                lines.append("- Type: map[string -> any]\n")
        table = _render_object_properties_table(schema, components)
        if table:
            lines.append(table)
        lines.append(_render_enum(schema))
        lines.append(_render_examples(schema))
        return "".join(lines) if any(lines) else render_schema(schema)

    # Arrays / primitives
    lines = [f"- **Type**: {_schema_type_repr(schema, components)}\n"]
    desc = str(schema.get("description", "")).strip()
    if desc:
        lines.append(f"- **Description**: {md_escape(desc)}\n")
    default = schema.get("default")
    if default is not None:
        try:
            default_str = json.dumps(default)
        except Exception:
            default_str = str(default)
        lines.append(f"- **Default**: {md_escape(default_str)}\n")
    lines.append(_render_enum(schema))
    lines.append(_render_examples(schema))
    return "".join(lines)


def schema_title(schema: Mapping[str, Any]) -> Optional[str]:
    title = schema.get("title")
    if isinstance(title, str) and title.strip():
        return title
    return None


def render_parameters_table(params: List[Mapping[str, Any]], components: Mapping[str, Any]) -> str:
    if not params:
        return ""
    rows: List[str] = ["| Name | In | Required | Type | Description |", "| --- | --- | :---: | --- | --- |"]
    for p in params:
        name = md_escape(str(p.get("name", "")))
        loc = md_escape(str(p.get("in", "")))
        required = "yes" if p.get("required", False) else "no"
        desc = md_escape(str(p.get("description", "")).strip())
        typ = ""
        schema = p.get("schema")
        if isinstance(schema, Mapping):
            if "$ref" in schema:
                ref_name, ref_schema = resolve_ref(str(schema["$ref"]), components)
                typ = ref_name
            else:
                typ = str(schema.get("type", ""))
                if schema.get("enum"):
                    try:
                        enum_vals = ", ".join(map(str, schema["enum"]))
                        typ = f"{typ} (enum: {enum_vals})"
                    except Exception:
                        pass
        rows.append(f"| {name} | {loc} | {required} | {md_escape(typ)} | {desc} |")
    return "\n".join(rows) + "\n\n"


def render_request_body(rb: Mapping[str, Any], components: Mapping[str, Any]) -> str:
    lines: List[str] = []
    required = rb.get("required", False)
    lines.append(f"- Required: {'yes' if required else 'no'}\n")
    content = rb.get("content", {})
    if not isinstance(content, Mapping):
        return "\n".join(lines) + "\n"
    for content_type, media in content.items():
        lines.append(f"- Content type: `{content_type}`\n")
        if not isinstance(media, Mapping):
            continue
        schema = media.get("schema")
        if schema is None:
            continue
        # Resolve top-level $ref for readability
        if isinstance(schema, Mapping) and "$ref" in schema:
            ref_name, ref_schema = resolve_ref(str(schema["$ref"]), components)
            lines.append(f"  - Schema: {_linkify_anchor(ref_name)}\n\n")
            if isinstance(ref_schema, Mapping):
                lines.append(render_schema_readable(ref_schema, components))
        else:
            lines.append(render_schema_readable(schema, components))
    return "\n".join(lines) + "\n"


def render_responses(responses: Mapping[str, Any], components: Mapping[str, Any]) -> str:
    out: List[str] = []
    for status, spec in sorted(responses.items(), key=lambda kv: kv[0]):
        out.append(f"#### {status}\n")
        if not isinstance(spec, Mapping):
            out.append("(no details)\n\n")
            continue
        desc = str(spec.get("description", "")).strip()
        if desc:
            out.append(desc + "\n\n")
        content = spec.get("content", {})
        if isinstance(content, Mapping):
            for content_type, media in content.items():
                out.append(f"- Content type: `{content_type}`\n")
                if not isinstance(media, Mapping):
                    continue
                schema = media.get("schema")
                if schema is None:
                    continue
                if isinstance(schema, Mapping) and "$ref" in schema:
                    ref_name, ref_schema = resolve_ref(str(schema["$ref"]), components)
                    out.append(f"  - Schema: {_linkify_anchor(ref_name)}\n\n")
                    if isinstance(ref_schema, Mapping):
                        out.append(render_schema_readable(ref_schema, components))
                else:
                    out.append(render_schema_readable(schema, components))
        out.append("\n")
    return "".join(out)


def build_index_md(openapi: Mapping[str, Any]) -> str:
    info = openapi.get("info", {}) if isinstance(openapi, Mapping) else {}
    title = str(info.get("title", "Workflows Server API")).strip() or "Workflows Server API"
    version = str(info.get("version", "")).strip()
    header = [
        "---",
        f"title: {title} Reference",
        "---",
        "",
        "<!-- THIS FILE IS GENERATED. DO NOT EDIT MANUALLY. -->",
        "",
    ]
    header.append(f"**Version**: `{version}`\n" if version else "")
    header.append(
        "This reference is generated from the server's OpenAPI schema. It lists all endpoints, their parameters, request bodies, and responses.\n\n"
    )

    # Paths
    paths = openapi.get("paths", {}) if isinstance(openapi, Mapping) else {}
    components = openapi.get("components", {}) if isinstance(openapi, Mapping) else {}

    # Order paths lexicographically for stability
    for path_str in sorted(paths.keys()):
        header.append(f"### {path_str}\n")
        methods = paths[path_str]
        if not isinstance(methods, Mapping):
            header.append("(no method definitions)\n\n")
            continue
        # Order methods by common HTTP verb order
        verb_order = {"get": 0, "post": 1, "put": 2, "patch": 3, "delete": 4, "options": 5, "head": 6}
        for method, op in sorted(methods.items(), key=lambda kv: (verb_order.get(kv[0].lower(), 99), kv[0])):
            if not isinstance(op, Mapping):
                continue
            verb = method.upper()
            summary = str(op.get("summary", "")).strip()
            description = str(op.get("description", "")).strip()
            header.append(f"#### {verb}\n")
            if summary:
                header.append(f"**Summary**: {md_escape(summary)}\n\n")
            if description:
                header.append(md_escape(description) + "\n\n")

            params = op.get("parameters", [])
            if isinstance(params, list) and params:
                header.append("**Parameters**\n\n")
                header.append(render_parameters_table(params, components))

            request_body = op.get("requestBody")
            if isinstance(request_body, Mapping):
                header.append("**Request Body**\n\n")
                header.append(render_request_body(request_body, components))

            responses = op.get("responses")
            if isinstance(responses, Mapping) and responses:
                header.append("**Responses**\n\n")
                header.append(render_responses(responses, components))

        header.append("\n")

    # Components appendix (schemas only)
    schemas = components.get("schemas", {}) if isinstance(components, Mapping) else {}
    if isinstance(schemas, Mapping) and schemas:
        header.append("### Components\n\n")
        header.append("These are the component schemas referenced above.\n\n")
        for name in sorted(schemas.keys()):
            header.append(f"#### {name}\n\n")
            schema_obj = schemas[name]
            if isinstance(schema_obj, Mapping):
                header.append(render_schema_readable(schema_obj, components))
            else:
                header.append(render_schema(schema_obj))
            header.append("\n")

    return "\n".join([line for line in header if line is not None])


def write_index(dest_dir: Path, content: str) -> None:
    ensure_out_dir(dest_dir)
    out_path = dest_dir / "index.md"
    # Ensure trailing newline to satisfy end-of-file linters
    if not content.endswith("\n"):
        content = content + "\n"
    out_path.write_text(content, encoding="utf-8")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Generate Markdown reference from OpenAPI JSON")
    parser.add_argument("--openapi", type=str, default=None, help="Path to OpenAPI json file")
    parser.add_argument("--out", type=str, required=True, help="Output directory for generated markdown")
    args = parser.parse_args()

    openapi_path = Path(args.openapi) if args.openapi is not None else None
    out_dir = Path(args.out)
    return Config(openapi_path=openapi_path, out_dir=out_dir)


def main() -> None:
    config = parse_args()
    openapi = load_openapi(config)
    content = build_index_md(openapi)
    write_index(config.out_dir, content)


if __name__ == "__main__":
    main()
