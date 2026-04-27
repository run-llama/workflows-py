# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Declarative column framework for tabular CLI read commands.

A "tabular read command" emits a CLI-side display model whose fields carry
:class:`Column` markers in their ``Annotated[]`` metadata. A small walker
(``resolve_columns``) reads the markers; ``render_columns`` derives a
plain-whitespace table; consumers compose this with ``render_output`` (in
``cli.options``) to dispatch text/json/yaml/wide.

The ``Annotated[]`` channel is intentionally open: future markers
(``YamlComment`` for pedagogical comments in template output, ``Alias`` for
legacy-name input tolerance on ``apply``, etc.) live alongside ``Column`` on
the same fields. Each consumer reads ``field.metadata`` and filters on its own
marker class via ``isinstance`` — markers do not register with the framework.

Per-command display models (``DeploymentDisplay``, ``ReleaseDisplay``,
``AuthProfileDisplay``, …) live with the commands that consume them and
import the primitives from this module.
"""

from __future__ import annotations

import functools
import types
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin

from pydantic import BaseModel

SECRET_MASK = "********"


@dataclass(frozen=True)
class Column:
    """Marker placed in a field's ``Annotated[]`` metadata to declare a column.

    The marker is a pure data class; it carries no behaviour. The walker
    (:func:`resolve_columns`) discovers ``Column`` instances and the renderer
    (:func:`render_columns`) consumes them.

    Args:
        header: Column header rendered verbatim in the table.
        format: Optional cell formatter. Called with the raw field value
            (only when non-None). Must return a string.
        default: Cell text when the value (or any nested-model parent on the
            field path) is ``None``.
        wide: When ``True``, the column appears only under ``-o wide``.
    """

    header: str
    format: Callable[[Any], str] | None = None
    default: str = ""
    wide: bool = False


@dataclass(frozen=True)
class ResolvedColumn:
    """A walker-derived column: its declaration path plus the marker."""

    path: tuple[str, ...]
    column: Column


def _is_basemodel(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _unwrap_optional_model(annotation: Any) -> type[BaseModel] | None:
    """If ``annotation`` is ``BaseModel``, ``Optional[BaseModel]`` or
    ``BaseModel | None``, return the model class. Otherwise ``None``."""

    if _is_basemodel(annotation):
        return annotation  # type: ignore[return-value]
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1 and _is_basemodel(non_none[0]):
            return non_none[0]
    return None


@functools.cache
def resolve_columns(model_cls: type[BaseModel]) -> tuple[ResolvedColumn, ...]:
    """Walk a display model in declaration order and return its columns.

    Field annotations carrying a ``Column`` marker yield a leaf column at the
    field's path. Fields whose annotation is a ``BaseModel`` (or
    ``Optional[BaseModel]``) are descended into. All other fields are skipped.

    A field carrying multiple ``Column`` markers is a typo: the walker raises
    ``ValueError`` rather than silently picking one.

    Display models are assumed to form a tree, not a graph — circular
    references are not supported.
    """

    return tuple(_walk(model_cls, ()))


def _walk(model_cls: type[BaseModel], prefix: tuple[str, ...]) -> list[ResolvedColumn]:
    out: list[ResolvedColumn] = []
    for name, info in model_cls.model_fields.items():
        path = prefix + (name,)
        cols = [m for m in info.metadata if isinstance(m, Column)]
        if len(cols) > 1:
            raise ValueError(
                f"{model_cls.__name__}.{name}: multiple Column annotations on a single field"
            )
        if cols:
            out.append(ResolvedColumn(path=path, column=cols[0]))
            continue
        nested = _unwrap_optional_model(info.annotation)
        if nested is not None:
            out.extend(_walk(nested, path))
    return out


def _extract_cell(row: BaseModel, column: ResolvedColumn) -> str:
    value: Any = row
    for part in column.path:
        if value is None:
            return column.column.default
        value = getattr(value, part)
    if value is None:
        return column.column.default
    if column.column.format is not None:
        return column.column.format(value)
    return str(value)


def render_columns(
    rows: list[BaseModel] | list[Any],
    *,
    wide: bool = False,
) -> None:
    """Render ``rows`` as a plain-whitespace table using ``Column`` metadata.

    ``rows`` must be homogeneous; the row class is read from the first
    element. An empty list still emits headers (matches ``render_table``'s
    empty-row behaviour). Columns marked ``wide=True`` are filtered out unless
    ``wide`` is ``True``.
    """

    from llama_agents.cli.render import render_table  # local: avoid cycle

    if not rows:
        # No row to derive a class from. Caller is expected to have emitted a
        # status message ("No X found") before reaching here — but if not,
        # we silently emit nothing rather than guessing the column layout.
        return

    row_cls = type(rows[0])
    if not isinstance(rows[0], BaseModel):
        raise TypeError(
            f"render_columns expects BaseModel rows; got {row_cls.__name__}"
        )

    cols = [c for c in resolve_columns(row_cls) if wide or not c.column.wide]
    columns = [(c.column.header, c.column.header) for c in cols]
    table_rows: list[dict[str, str]] = []
    for row in rows:
        table_rows.append({c.column.header: _extract_cell(row, c) for c in cols})
    render_table(table_rows, columns)
