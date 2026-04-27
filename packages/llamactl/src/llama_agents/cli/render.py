# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Plain-whitespace table renderer for llamactl.

No colors, no truncation, no Rich. Long values let the terminal wrap. The
helper exists so every read command emits the same simple, grep-friendly
shape.
"""

from __future__ import annotations

from datetime import datetime, timezone

import click

GH_PREFIX = "https://github.com/"


def gh_short(repo_url: str) -> str:
    """Return ``gh:org/repo`` for github URLs, otherwise the input unchanged.

    Used in table cells only; YAML/JSON keep the full URL.
    """
    if repo_url.startswith(GH_PREFIX):
        return "gh:" + repo_url.removeprefix(GH_PREFIX)
    return repo_url


def short_sha(sha: str) -> str:
    """Return the first 7 characters of ``sha``, or the input unchanged.

    Used in table cells; JSON/YAML keep the full SHA. Safe on shorter input.
    """
    return sha[:7] if len(sha) > 7 else sha


def star_marker(active: bool) -> str:
    """Render a boolean as ``"*"`` (true) or ``""`` (false).

    Used by the ``ACTIVE`` column on profile/env/org tables.
    """
    return "*" if active else ""


def format_iso_z(dt: datetime) -> str:
    """Format ``dt`` as a UTC ISO 8601 string with a ``Z`` suffix.

    Tz-aware datetimes are converted to UTC. Naive datetimes are assumed to
    already be UTC (the server emits UTC; this is a safety fallback rather
    than an invitation to pass local time). Fractional seconds are dropped:
    current data doesn't carry them and the table cell stays narrow.

    The output (``YYYY-MM-DDTHH:MM:SSZ``) matches what Pydantic emits in
    JSON mode, so text and JSON encodings of the same instant agree.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def render_table(
    rows: list[dict[str, str]],
    columns: list[tuple[str, str]],
) -> None:
    """Render a plain whitespace table to stdout.

    Args:
        rows: Each row is a ``{column_key: cell_value}`` mapping. Cells must
            already be strings (callers handle ``None`` → ``"-"``).
        columns: Ordered ``(header, key)`` pairs. Headers are rendered
            verbatim — callers pass uppercase headers if they want them.

    Column widths are sized to the widest cell (header included) plus a
    two-space gutter on the right except the last column. No truncation.
    """
    widths: list[int] = [
        max(len(header), *(len(row.get(key, "")) for row in rows))
        if rows
        else len(header)
        for header, key in columns
    ]

    def _format_line(values: list[str]) -> str:
        parts: list[str] = []
        last = len(values) - 1
        for i, value in enumerate(values):
            if i == last:
                parts.append(value)
            else:
                parts.append(value.ljust(widths[i] + 2))
        return "".join(parts).rstrip()

    headers = [header for header, _ in columns]
    click.echo(_format_line(headers))
    for row in rows:
        click.echo(_format_line([row.get(key, "") for _, key in columns]))
