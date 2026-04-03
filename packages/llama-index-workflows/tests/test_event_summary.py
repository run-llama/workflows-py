# ty: ignore[unknown-argument]
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import pytest
from workflows._event_summary import summarize_event
from workflows.events import Event, StartEvent, StopEvent


class RetrievalEvent(Event):
    query: str
    top_k: int = 5


class CustomStopEvent(StopEvent):
    summary: str = ""


@pytest.fixture
def long_string() -> str:
    return "a" * 200


@pytest.fixture
def large_list() -> list[int]:
    return list(range(20))


@pytest.fixture
def large_dict() -> dict[str, int]:
    return {f"key_{i}": i for i in range(15)}


def test_simple_start_event_with_kwargs() -> None:
    ev = StartEvent(topic="Pirates")  # type: ignore[unknown-argument]
    result = summarize_event(ev)
    assert result == "StartEvent(topic='Pirates')"


def test_stop_event_includes_result() -> None:
    ev = StopEvent(result="hello")
    result = summarize_event(ev)
    assert "result=" in result
    assert "hello" in result
    assert result.startswith("StopEvent(")


def test_custom_event_with_pydantic_fields() -> None:
    ev = RetrievalEvent(query="what is the meaning of life?", top_k=5)
    result = summarize_event(ev)
    assert result.startswith("RetrievalEvent(")
    assert "query=" in result
    assert "top_k=5" in result


def test_long_string_value_truncated(long_string: str) -> None:
    ev = StartEvent(content=long_string)  # type: ignore[unknown-argument]
    result = summarize_event(ev)
    # The full 200-char string should not appear
    assert long_string not in result
    # Should contain truncation indicator
    assert "..." in result


def test_large_list_shows_item_count(large_list: list[int]) -> None:
    ev = StartEvent(items=large_list)  # type: ignore[unknown-argument]
    result = summarize_event(ev)
    assert "[20 items]" in result


def test_large_dict_shows_key_count(large_dict: dict[str, int]) -> None:
    ev = StartEvent(mapping=large_dict)  # type: ignore[unknown-argument]
    result = summarize_event(ev)
    assert "{15 keys}" in result


def test_overall_output_truncated_to_max_length() -> None:
    ev = RetrievalEvent(
        query="a very long query string that keeps going and going",
        top_k=10,
        extra_field_1="some data",  # type: ignore[unknown-argument]
        extra_field_2="more data",  # type: ignore[unknown-argument]
        extra_field_3="even more data",  # type: ignore[unknown-argument]
    )
    max_len = 50
    result = summarize_event(ev, max_length=max_len)
    assert len(result) <= max_len
    assert result.endswith("...")


def test_mixed_pydantic_fields_and_data_entries() -> None:
    ev = RetrievalEvent(query="hello world", top_k=3, source="wikipedia", page=7)  # type: ignore[unknown-argument]
    result = summarize_event(ev)
    assert "query='hello world'" in result
    assert "top_k=3" in result
    assert "source=" in result
    assert "page=" in result


def test_stop_event_subclass_includes_result() -> None:
    ev = CustomStopEvent(result=42, summary="done")  # type: ignore[unknown-argument]
    result = summarize_event(ev)
    assert result.startswith("CustomStopEvent(")
    assert "result=" in result
    assert "summary=" in result


def test_stop_event_with_none_result() -> None:
    ev = StopEvent()
    result = summarize_event(ev)
    assert result == "StopEvent()"
    assert "result=" not in result


def test_empty_event() -> None:
    ev = StartEvent()
    result = summarize_event(ev)
    assert result == "StartEvent()"


def test_small_list_shown_inline() -> None:
    ev = StartEvent(items=[1, 2, 3])  # type: ignore[unknown-argument]
    result = summarize_event(ev)
    # Small lists should be shown inline, not as "[3 items]"
    assert "[1, 2, 3]" in result or "items=" in result


def test_small_dict_shown_inline() -> None:
    ev = StartEvent(payload={"a": 1})  # type: ignore[unknown-argument]
    result = summarize_event(ev)
    assert "payload=" in result
