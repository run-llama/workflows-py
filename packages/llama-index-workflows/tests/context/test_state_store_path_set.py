# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Copy-on-write behavior of `StateStore.set` and `set_by_path_copy`.

`set` rebuilds only the containers along the written path. These tests pin the
two things that buys — values off the path keep their identity and are never
copied — and the parity with the in-place `set_by_path` it replaced.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Callable

import pytest
from pydantic import BaseModel
from workflows.context.serializers import JsonSerializer
from workflows.context.state_store import (
    MAX_DEPTH,
    DictState,
    InMemoryStateStore,
    StateStoreFacade,
    get_by_path,
    set_by_path,
    set_by_path_copy,
)
from workflows.events import DictLikeModel, StopEvent

from .test_state_store_facade import FakeDurableStorage


class TypedRoot(BaseModel):
    a: int = 0
    b: int = 0


class TypedDictLike(DictLikeModel):
    foo: int = 7


class ReadOnlyAttr:
    """Path container whose attribute cannot be assigned."""

    @property
    def value(self) -> int:
        return 1


class Uncopyable:
    """Live handle that refuses to be copied, like a lock or socket wrapper."""

    def __init__(self, target: Any = None) -> None:
        self.slot = 0
        self.target = target

    def __copy__(self) -> Uncopyable:
        raise TypeError("cannot copy a live handle")

    def __deepcopy__(self, memo: dict[int, Any]) -> Uncopyable:
        raise TypeError("cannot copy a live handle")


class CopyCounter:
    """Value that records every attempt to copy it, and shares itself instead."""

    def __init__(self) -> None:
        self.copies = 0

    def __copy__(self) -> CopyCounter:
        self.copies += 1
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> CopyCounter:
        self.copies += 1
        return self


def durable_store() -> StateStoreFacade[DictState]:
    return StateStoreFacade(FakeDurableStorage(), DictState, JsonSerializer())


def make_store(durable: bool) -> Any:
    return durable_store() if durable else InMemoryStateStore(DictState())


def dump(value: Any) -> Any:
    """Plain-data view of a state graph, for comparing two writers' results."""
    if isinstance(value, DictLikeModel):
        return {
            "fields": {k: dump(getattr(value, k)) for k in type(value).model_fields},
            "data": {k: dump(v) for k, v in value.items()},
        }
    if isinstance(value, BaseModel):
        return {k: dump(getattr(value, k)) for k in type(value).model_fields}
    if isinstance(value, dict):
        return {k: dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump(v) for v in value]
    return value


# ============================================================================
# Readers hold committed state across a write
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_reader_holding_state_is_unaffected_by_set(durable: bool) -> None:
    """Values read before a write keep their old contents after it commits."""
    store = make_store(durable)
    await store.set("a.b", 1)

    snapshot = await store.get_state()
    nested = await store.get("a")

    await store.set("a.b", 2)

    assert snapshot["a"]["b"] == 1
    assert nested["b"] == 1
    assert await store.get("a.b") == 2


@pytest.mark.asyncio
async def test_nested_set_rebuilds_only_the_spine() -> None:
    """Containers on the path are new objects; everything else is shared."""
    store: InMemoryStateStore[DictState] = InMemoryStateStore(DictState())
    await store.set("a", {"b": {"c": 0}, "sibling": {"keep": 1}})
    await store.set("other", {"unrelated": 1})

    before = await store.get_state()
    old_a, old_b = before["a"], before["a"]["b"]
    old_sibling, old_other = before["a"]["sibling"], before["other"]

    await store.set("a.b.c", 1)
    after = await store.get_state()

    assert after["a"] is not old_a
    assert after["a"]["b"] is not old_b
    assert after["a"]["sibling"] is old_sibling
    assert after["other"] is old_other
    assert old_b["c"] == 0
    assert after["a"]["b"]["c"] == 1


@pytest.mark.asyncio
async def test_set_does_not_copy_values_off_the_path() -> None:
    """Repeated writes never copy an unrelated value, however large."""
    store: InMemoryStateStore[DictState] = InMemoryStateStore(DictState())
    sentinel = CopyCounter()
    await store.set("sentinel", sentinel)
    await store.set("bulk", {str(i): i for i in range(100_000)})
    bulk = await store.get("bulk")

    for i in range(200):
        await store.set("counter", i)

    assert sentinel.copies == 0
    assert await store.get("sentinel") is sentinel
    assert await store.get("bulk") is bulk
    assert await store.get("counter") == 199


@pytest.mark.asyncio
async def test_concurrent_sets_do_not_lose_writes() -> None:
    """Read-modify-write under the lock keeps every concurrent write."""
    store: InMemoryStateStore[DictState] = InMemoryStateStore(DictState())
    await store.set("root", {})

    async def writer(i: int) -> None:
        await store.set(f"k{i}", i)
        await store.set(f"root.n{i}", i)

    await asyncio.gather(*(writer(i) for i in range(50)))

    state = await store.get_state()
    for i in range(50):
        assert state[f"k{i}"] == i
        assert state["root"][f"n{i}"] == i


@pytest.mark.asyncio
async def test_uncopyable_container_on_path_is_written_through() -> None:
    """A live handle on the path still accepts a write, in place."""
    store: InMemoryStateStore[DictState] = InMemoryStateStore(DictState())
    live = Uncopyable()
    await store.set("live", live)

    await store.set("live.slot", 7)

    assert await store.get("live") is live
    assert await store.get("live.slot") == 7


# ============================================================================
# Parity with the in-place writer
# ============================================================================

PARITY_CASES: list[tuple[str, Callable[[], Any], str, Any]] = [
    ("missing_intermediates", DictState, "x.y.z", 1),
    ("method_named_segment", DictState, "items.nested", 1),
    ("numeric_string_key", DictState, "0", "zero"),
    ("list_index", lambda: DictState(nums=[1, 2, 3]), "nums.0", 9),
    ("list_index_out_of_range", lambda: DictState(nums=[1]), "nums.5", 9),
    ("declared_field", TypedDictLike, "foo", 9),
    ("property_segment", lambda: DictState(ev=StopEvent(result=42)), "ev.result", 7),
    ("typed_root_field", TypedRoot, "a", 5),
    ("typed_root_missing_field", TypedRoot, "nope", 5),
    ("through_a_scalar", lambda: DictState(n=5), "n.x", 1),
    ("through_a_tuple", lambda: DictState(t=(1, 2)), "t.0", 9),
    ("read_only_attribute", lambda: DictState(obj=ReadOnlyAttr()), "obj.value", 5),
    ("nested_existing", lambda: DictState(a={"b": {"c": 0}}), "a.b.c", 1),
]


@pytest.mark.parametrize(
    ("build", "path", "value"),
    [pytest.param(b, p, v, id=name) for name, b, p, v in PARITY_CASES],
)
def test_set_by_path_copy_matches_set_by_path(
    build: Callable[[], Any], path: str, value: Any
) -> None:
    """Same result, or the same failure with state left alone."""
    in_place = build()
    in_place_error: type[BaseException] | None = None
    try:
        set_by_path(in_place, path, value)
    except Exception as exc:
        in_place_error = type(exc)

    copied_from = build()
    untouched = dump(copied_from)
    copy_error: type[BaseException] | None = None
    result: Any = None
    try:
        result = set_by_path_copy(copied_from, path, value)
    except Exception as exc:
        copy_error = type(exc)

    assert copy_error is in_place_error
    if copy_error is not None:
        assert dump(copied_from) == untouched
        return
    assert dump(result) == dump(in_place)
    assert dump(copied_from) == untouched


def test_declared_field_is_not_shadowed_in_data() -> None:
    """Declared fields stay fields on the rebuilt copy."""
    result = set_by_path_copy(TypedDictLike(), "foo", 9)
    assert result.foo == 9
    assert "foo" not in result._data


def test_set_by_path_copy_shares_values_off_the_path() -> None:
    """Only the path is rebuilt; sibling values keep their identity."""
    sibling = {"big": list(range(10))}
    state = DictState(a={"b": 0}, sibling=sibling)

    result = set_by_path_copy(state, "a.b", 1)

    assert result is not state
    assert result["sibling"] is sibling
    assert result["a"] is not state["a"]
    assert state["a"]["b"] == 0


# ============================================================================
# Path validation
# ============================================================================


def test_empty_path_raises() -> None:
    with pytest.raises(ValueError):
        set_by_path_copy(DictState(), "", 1)


def test_max_depth_boundary() -> None:
    """MAX_DEPTH segments write; one more raises, as with the in-place writer."""
    path = ".".join(f"s{i}" for i in range(MAX_DEPTH))
    state = set_by_path_copy(DictState(), path, 1)
    assert get_by_path(state, path) == 1

    with pytest.raises(ValueError):
        set_by_path_copy(DictState(), f"{path}.over", 1)


@pytest.mark.asyncio
async def test_set_inside_edit_state_raises_for_any_path() -> None:
    """Path validation must not preempt the nested-writer check."""
    store: InMemoryStateStore[DictState] = InMemoryStateStore(DictState())

    with pytest.raises(RuntimeError):
        async with store.edit_state():
            await store.set("a", 1)

    with pytest.raises(RuntimeError):
        async with store.edit_state():
            await store.set("", 1)


# ============================================================================
# Durable backends
# ============================================================================


@pytest.mark.asyncio
async def test_durable_set_writes_one_row_and_keeps_siblings() -> None:
    """Durable writes still re-encode the whole row, once per set."""
    storage = FakeDurableStorage()
    store: StateStoreFacade[DictState] = StateStoreFacade(
        storage, DictState, JsonSerializer()
    )
    await store.set("a", 1)
    await store.set("b", {"c": 2})
    saves = storage.save_count

    await store.set("b.c", 3)

    assert storage.save_count == saves + 1
    assert await store.get("a") == 1
    assert await store.get("b.c") == 3


@pytest.mark.asyncio
async def test_edit_state_still_isolates_the_whole_state() -> None:
    """The transactional block keeps its copy semantics."""
    store: InMemoryStateStore[DictState] = InMemoryStateStore(DictState())
    await store.set("nums", [1])

    async with store.edit_state() as state:
        state["nums"].append(2)
        assert await store.get("nums") == [1]

    assert await store.get("nums") == [1, 2]


def test_deepcopy_of_state_is_unaffected_by_sharing() -> None:
    """A caller that wants isolation can still deep-copy the state itself."""
    state = DictState(a={"b": 1})
    isolated = deepcopy(state)
    result = set_by_path_copy(state, "a.b", 2)

    assert isolated["a"]["b"] == 1
    assert result["a"]["b"] == 2


class SelfCopy:
    """Path container that opts out of copying by returning itself."""

    def __init__(self) -> None:
        self.slot = 0

    def __copy__(self) -> SelfCopy:
        return self


class CountingSetter:
    """Path container whose attribute assignment has a visible side effect."""

    writes: list[Any]
    slot: int

    def __init__(self, writes: list[Any]) -> None:
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "slot", 0)

    def __setattr__(self, name: str, value: Any) -> None:
        self.writes.append(value)
        object.__setattr__(self, name, value)


def test_container_that_copies_to_itself_is_written_through() -> None:
    """A `__copy__` returning self means "share me", so write in place."""
    shared = SelfCopy()
    state = DictState(shared=shared)

    result = set_by_path_copy(state, "shared.slot", 1)

    assert result is state
    assert shared.slot == 1


def test_uncopyable_ancestor_assigns_exactly_once() -> None:
    """Falling back must not repeat assignments the rebuild already made."""
    writes: list[Any] = []
    live = Uncopyable(CountingSetter(writes))
    state = DictState(live=live)

    result = set_by_path_copy(state, "live.target.slot", 7)

    assert result is state
    assert live.target.slot == 7
    assert writes == [7]
