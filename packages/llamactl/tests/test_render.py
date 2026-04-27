# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for the plain-whitespace table renderer."""

from __future__ import annotations

from datetime import datetime, timezone

import click
from click.testing import CliRunner
from llama_agents.cli.render import format_iso_z, gh_short, render_table


def _capture(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    @click.command()
    def _cmd() -> None:
        render_table(rows, columns)

    result = CliRunner().invoke(_cmd, [])
    assert result.exit_code == 0, result.output
    return result.output


def test_render_table_emits_headers_and_rows() -> None:
    output = _capture(
        rows=[
            {"name": "alpha", "phase": "Running"},
            {"name": "beta", "phase": "Suspended"},
        ],
        columns=[("NAME", "name"), ("PHASE", "phase")],
    )
    lines = output.splitlines()
    assert lines[0].startswith("NAME")
    assert "PHASE" in lines[0]
    assert "alpha" in lines[1]
    assert "Running" in lines[1]
    assert "beta" in lines[2]
    assert "Suspended" in lines[2]


def test_render_table_no_ansi_escapes_or_truncation() -> None:
    long_value = "https://github.com/run-llama/template-workflow-classify-extract-sec"
    output = _capture(
        rows=[{"name": "a", "repo": long_value}],
        columns=[("NAME", "name"), ("REPO", "repo")],
    )
    assert "\x1b[" not in output
    assert "…" not in output
    assert long_value in output


def test_render_table_column_width_uses_widest_cell() -> None:
    output = _capture(
        rows=[
            {"name": "short", "phase": "Pending"},
            {"name": "much-longer-name", "phase": "RollingOut"},
        ],
        columns=[("NAME", "name"), ("PHASE", "phase")],
    )
    lines = output.splitlines()
    # Header column 2 starts at the same column as data column 2.
    header_phase_start = lines[0].index("PHASE")
    row1_phase_start = lines[1].index("Pending")
    row2_phase_start = lines[2].index("RollingOut")
    assert header_phase_start == row1_phase_start == row2_phase_start


def test_render_table_handles_empty_rows() -> None:
    output = _capture(rows=[], columns=[("NAME", "name"), ("PHASE", "phase")])
    # Header row only; trailing whitespace stripped.
    assert output.strip().split("\n") == ["NAME  PHASE"]


def test_gh_short_translates_github_urls() -> None:
    assert (
        gh_short("https://github.com/run-llama/template-workflow")
        == "gh:run-llama/template-workflow"
    )


def test_gh_short_passes_through_non_github() -> None:
    assert gh_short("https://gitlab.com/x/y") == "https://gitlab.com/x/y"
    assert gh_short("internal://repo") == "internal://repo"


def test_format_iso_z_tz_aware_utc() -> None:
    dt = datetime(2026, 4, 25, 15, 1, 15, tzinfo=timezone.utc)
    assert format_iso_z(dt) == "2026-04-25T15:01:15Z"


def test_format_iso_z_tz_aware_non_utc_is_converted() -> None:
    # Pacific (UTC-8 standard time): 07:01:15-08:00 == 15:01:15Z
    from datetime import timedelta
    from datetime import timezone as _tz

    pst = _tz(timedelta(hours=-8))
    dt = datetime(2026, 4, 25, 7, 1, 15, tzinfo=pst)
    assert format_iso_z(dt) == "2026-04-25T15:01:15Z"


def test_format_iso_z_naive_is_treated_as_utc() -> None:
    # Documented behavior: naive datetimes are assumed to already be UTC.
    dt = datetime(2026, 4, 25, 15, 1, 15)
    assert format_iso_z(dt) == "2026-04-25T15:01:15Z"
