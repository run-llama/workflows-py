# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for the declarative ``Column`` framework in ``cli.display``."""

from __future__ import annotations

from typing import Any, Literal

import pytest
from llama_agents.cli.display import (
    Column,
    render_columns,
    resolve_columns,
)
from pydantic import BaseModel
from typing_extensions import Annotated


class _Flat(BaseModel):
    name: Annotated[str, Column("NAME")]
    note: str  # no Column → skipped
    age: Annotated[int, Column("AGE", format=lambda v: f"~{v}")]


class _Inner(BaseModel):
    phase: Annotated[str, Column("PHASE")]
    secret: str  # no Column → skipped


class _Nested(BaseModel):
    name: Annotated[str, Column("NAME")]
    inner: _Inner
    optional_inner: _Inner | None = None


class _Wide(BaseModel):
    name: Annotated[str, Column("NAME")]
    extra: Annotated[str, Column("EXTRA", wide=True)] = "x"


def test_resolve_columns_flat_model_declaration_order() -> None:
    cols = resolve_columns(_Flat)
    assert [c.column.header for c in cols] == ["NAME", "AGE"]
    assert [c.path for c in cols] == [("name",), ("age",)]


def test_resolve_columns_descends_nested_models() -> None:
    cols = resolve_columns(_Nested)
    # Outer NAME, then inner PHASE (descended), then optional_inner PHASE.
    assert [c.column.header for c in cols] == ["NAME", "PHASE", "PHASE"]
    assert [c.path for c in cols] == [
        ("name",),
        ("inner", "phase"),
        ("optional_inner", "phase"),
    ]


def test_resolve_columns_skips_field_without_column() -> None:
    cols = resolve_columns(_Inner)
    # ``secret`` carries no Column → excluded.
    assert [c.column.header for c in cols] == ["PHASE"]


def test_resolve_columns_supports_multiple_independent_markers() -> None:
    """Forward-compat: extra markers on the same field don't perturb output."""

    class _Marker:
        pass

    class _M(BaseModel):
        name: Annotated[str, Column("NAME"), _Marker()]

    cols = resolve_columns(_M)
    assert len(cols) == 1
    assert cols[0].column.header == "NAME"


def test_resolve_columns_rejects_duplicate_columns_on_one_field() -> None:
    class _Bad(BaseModel):
        name: Annotated[str, Column("A"), Column("B")]

    with pytest.raises(ValueError, match="multiple Column"):
        resolve_columns(_Bad)


def test_render_columns_filters_wide(capsys: Any) -> None:
    rows = [_Wide(name="a"), _Wide(name="b", extra="z")]
    render_columns(rows)
    out = capsys.readouterr().out
    assert "EXTRA" not in out
    assert "NAME" in out

    render_columns(rows, wide=True)
    out = capsys.readouterr().out
    assert "EXTRA" in out
    assert "z" in out


def test_render_columns_applies_format_and_default(capsys: Any) -> None:
    class _M(BaseModel):
        ref: Annotated[str | None, Column("REF", default="-")] = None
        age: Annotated[int, Column("AGE", format=lambda v: f"~{v}")] = 0

    render_columns([_M(ref=None, age=3), _M(ref="main", age=7)])
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert "REF" in lines[0]
    # First row uses the default; format is applied to age.
    assert "-" in lines[1]
    assert "~3" in lines[1]
    assert "main" in lines[2]
    assert "~7" in lines[2]


def test_render_columns_propagates_none_through_missing_nested_model(
    capsys: Any,
) -> None:
    class _Inner2(BaseModel):
        phase: Annotated[str, Column("PHASE", default="-")]

    class _Outer(BaseModel):
        name: Annotated[str, Column("NAME")]
        inner: _Inner2 | None = None

    render_columns([_Outer(name="a", inner=None)])
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert "PHASE" in lines[0]
    # Missing nested model → cell renders the column's default.
    assert "-" in lines[1]


def test_resolve_columns_handles_optional_basemodel_union() -> None:
    cols = resolve_columns(_Nested)
    paths = {c.path for c in cols}
    assert ("optional_inner", "phase") in paths


def test_resolve_columns_is_cached() -> None:
    """Cache hit returns the same tuple instance."""
    a = resolve_columns(_Flat)
    b = resolve_columns(_Flat)
    assert a is b


def test_render_columns_literal_field_renders_value(capsys: Any) -> None:
    class _M(BaseModel):
        kind: Annotated[Literal["a", "b"], Column("KIND")] = "a"

    render_columns([_M()])
    out = capsys.readouterr().out
    assert "a" in out
