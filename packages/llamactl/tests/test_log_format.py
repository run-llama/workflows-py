# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Tests for the structlog body parser and plain renderer.

Phase 2 of the llamactl Slice A redesign.
"""

from __future__ import annotations

from llama_agents.cli.log_format import parse_log_body, render_plain


def test_parse_plain_string_passes_through() -> None:
    parsed = parse_log_body("plain stdout line")
    assert parsed.structured is False
    assert parsed.event == "plain stdout line"
    assert render_plain(parsed) == "plain stdout line"


def test_parse_invalid_json_passes_through() -> None:
    parsed = parse_log_body("{not json")
    assert parsed.structured is False
    assert render_plain(parsed) == "{not json"


def test_parse_dict_without_event_passes_through() -> None:
    parsed = parse_log_body('{"level": "info"}')
    assert parsed.structured is False


def test_parse_structlog_extracts_fields() -> None:
    line = (
        '{"event": "request done", "level": "info", '
        '"timestamp": "2026-04-26T12:34:56.789Z", '
        '"logger": "app.api", "request_id": "abc", "duration_ms": 42}'
    )
    parsed = parse_log_body(line)
    assert parsed.structured is True
    assert parsed.event == "request done"
    assert parsed.level == "info"
    assert parsed.logger == "app.api"
    assert parsed.request_id == "abc"
    assert parsed.extras == {"duration_ms": 42}


def test_render_plain_structured_layout() -> None:
    parsed = parse_log_body(
        '{"event": "ok", "level": "warning", '
        '"timestamp": "2026-04-26T12:34:56.000Z", "logger": "x"}'
    )
    rendered = render_plain(parsed)
    # Time portion only (not the full ISO date)
    assert "12:34:56.000" in rendered
    assert "WARNING" in rendered
    assert "x" in rendered
    assert "ok" in rendered
    # No date prefix
    assert "2026-04-26" not in rendered
