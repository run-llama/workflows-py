# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""Copy semantics of the state handed to an ``edit_state`` block."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from workflows.context.state_store import DictState, copy_state_for_edit
from workflows.events import Event


class Handle:
    """Stand-in for a live handle: a client, an engine, a tokenizer."""


class Undeepcopyable:
    """A live handle that raises on ``deepcopy``, like a lock or a module."""

    def __deepcopy__(self, memo: dict[int, Any]) -> Undeepcopyable:
        raise TypeError("cannot pickle this object")


class Memoryish(BaseModel):
    """Shape of an agent memory: data next to declared live handles."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[str] = Field(default_factory=list)
    tokenizer: Any = Field(default=None, exclude=True)
    blocks: list[Blockish] = Field(default_factory=list)


class Blockish(BaseModel):
    """A nested, non-excluded model that owns a handle of its own."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    facts: list[str] = Field(default_factory=list)
    llm: Any = Field(default=None, exclude=True)


class Node(BaseModel):
    """Self-referencing model, for cycles."""

    name: str
    peer: Node | None = None


class Pair(BaseModel):
    left: list[int] = Field(default_factory=list)
    right: list[int] = Field(default_factory=list)


class Privateer(BaseModel):
    """Data in a field, a live handle in a private attribute."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nums: list[int] = Field(default_factory=list)
    _engine: Any = PrivateAttr(default=None)


class TypedState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nums: list[int] = Field(default_factory=list)
    client: Any = Field(default=None, exclude=True)


@dataclass
class Record:
    nums: list[int] = field(default_factory=list)


def test_excluded_fields_are_shared_and_the_rest_is_isolated() -> None:
    tokenizer = Handle()
    memory = Memoryish(messages=["hi"], tokenizer=tokenizer)
    copied = copy_state_for_edit(DictState(memory=memory))["memory"]

    assert copied.tokenizer is tokenizer
    assert copied.messages == ["hi"]
    assert copied.messages is not memory.messages
    copied.messages.append("bye")
    assert memory.messages == ["hi"]


def test_nested_models_recurse_with_the_same_rule() -> None:
    llm = Handle()
    memory = Memoryish(blocks=[Blockish(facts=["a"], llm=llm)])
    copied = copy_state_for_edit(DictState(memory=memory))["memory"]

    block = copied.blocks[0]
    assert block is not memory.blocks[0]
    assert block.llm is llm
    block.facts.append("b")
    assert memory.blocks[0].facts == ["a"]


def test_model_keeps_its_identity_and_validation_state() -> None:
    memory = Memoryish(messages=["hi"], tokenizer=Handle())
    copied = copy_state_for_edit(DictState(memory=memory))["memory"]

    assert type(copied) is Memoryish
    assert copied.__pydantic_fields_set__ == memory.__pydantic_fields_set__
    assert copied.model_dump() == memory.model_dump()


def test_private_attributes_holding_data_stay_isolated() -> None:
    """``DictLikeModel`` keeps dynamic entries in a private attr — real data."""
    event = Event(payload={"a": 1})
    state = DictState(event=event)

    copied = copy_state_for_edit(state)["event"]
    copied["payload"]["a"] = 2

    assert event["payload"] == {"a": 1}


def test_private_attribute_handles_are_shared_without_losing_siblings() -> None:
    """A lazily-built engine cannot be copied, and must not be dropped either."""
    engine = Undeepcopyable()
    value = Privateer(nums=[1])
    value._engine = engine

    copied = copy_state_for_edit(DictState(value=value))["value"]

    assert copied._engine is engine
    copied.nums.append(2)
    assert value.nums == [1]


def test_cycles_terminate_and_mirror_the_original_shape() -> None:
    a = Node(name="a")
    b = Node(name="b", peer=a)
    a.peer = b

    copied = copy_state_for_edit(DictState(node=a))["node"]

    assert copied is not a
    assert copied.peer.peer is copied
    assert copied.name == "a"


def test_objects_referenced_twice_stay_one_object() -> None:
    shared = [1]
    pair = Pair()
    pair.left = shared
    pair.right = shared

    copied = copy_state_for_edit(DictState(pair=pair))["pair"]

    assert copied.left is copied.right
    assert copied.left is not shared


def test_the_same_model_under_two_state_keys_is_copied_once() -> None:
    memory = Memoryish(messages=["hi"])
    copied = copy_state_for_edit(DictState(a=memory, b=memory))

    assert copied["a"] is copied["b"]
    assert copied["a"] is not memory


def test_non_model_values_are_deep_copied() -> None:
    record = Record(nums=[1])
    copied = copy_state_for_edit(DictState(record=record, plain={"k": [1]}))

    assert copied["record"] is not record
    copied["record"].nums.append(2)
    copied["plain"]["k"].append(2)
    assert record.nums == [1]


def test_non_deepcopyable_values_are_kept_by_reference() -> None:
    """Regression for issues 709/710: an edit must not crash on a live handle."""
    client = Undeepcopyable()
    copied = copy_state_for_edit(DictState(client=client, nums=[1]))

    assert copied["client"] is client
    copied["nums"].append(2)


def test_a_non_deepcopyable_field_no_longer_costs_its_siblings() -> None:
    """Only the offending field is shared; declared data is still isolated."""

    class Client(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        lock: Any = None
        calls: list[str] = Field(default_factory=list)

    lock = Undeepcopyable()
    client = Client(lock=lock, calls=["a"])

    copied = copy_state_for_edit(DictState(client=client))["client"]

    assert copied.lock is lock
    copied.calls.append("b")
    assert client.calls == ["a"]


def test_a_handle_inside_a_container_does_not_cost_its_neighbors() -> None:
    """Containers are walked per element, so one bad entry is shared alone."""
    client = Undeepcopyable()
    copied = copy_state_for_edit(DictState(items=[client, [1]], by_key={"c": client}))

    assert copied["items"][0] is client
    assert copied["by_key"]["c"] is client
    copied["items"][1].append(2)
    assert copied["items"][1] == [1, 2]


def test_a_model_with_its_own_deepcopy_keeps_it() -> None:
    """`__deepcopy__` is the standard opt-out; the field-wise walk defers to it."""

    class SelfSharing(BaseModel):
        nums: list[int] = Field(default_factory=list)

        def __deepcopy__(self, memo: dict[int, Any] | None = None) -> SelfSharing:
            return self

    value = SelfSharing(nums=[1])
    assert copy_state_for_edit(DictState(value=value))["value"] is value


def test_typed_state_follows_the_same_rule() -> None:
    client = Handle()
    state = TypedState(nums=[1], client=client)

    copied = copy_state_for_edit(state)

    assert copied.client is client
    copied.nums.append(2)
    assert state.nums == [1]


REGISTRY: dict[str, RegistryEncoding] = {}


class RegistryEncoding:
    """Copy-cost model of a tiktoken ``Encoding``.

    Pickles by reference while it is the registry instance, and rebuilds from
    scratch once a copy has detached it — which is what turns one deep copy of
    an agent memory into an 80 ms BPE rebuild on every following copy.
    """

    rebuilds = 0

    def __init__(self, name: str) -> None:
        self.name = name

    def __deepcopy__(self, memo: dict[int, Any]) -> RegistryEncoding:
        if REGISTRY.get(self.name) is not self:
            RegistryEncoding.rebuilds += 1
        clone = RegistryEncoding.__new__(RegistryEncoding)
        clone.name = self.name
        return clone


@pytest.fixture
def encoding() -> Iterator[RegistryEncoding]:
    enc = RegistryEncoding("test-encoding")
    REGISTRY[enc.name] = enc
    RegistryEncoding.rebuilds = 0
    yield enc
    REGISTRY.pop(enc.name, None)


def test_repeated_copies_of_a_declared_tokenizer_never_rebuild_it(
    encoding: RegistryEncoding,
) -> None:
    """The reported regression: each copy detaches the tokenizer, so the next
    one rebuilds it. An excluded field is shared, so nothing ever detaches."""
    state = DictState(memory=Memoryish(messages=["hi"], tokenizer=encoding))

    for _ in range(5):
        state = copy_state_for_edit(state)
        assert state["memory"].tokenizer is encoding

    assert RegistryEncoding.rebuilds == 0


def test_an_undeclared_tokenizer_still_shows_the_pathology(
    encoding: RegistryEncoding,
) -> None:
    """Control: without the marker there is nothing to go on, and the rebuilds
    are back. This is what makes the test above meaningful."""

    class Undeclared(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        tokenizer: Any = None

    state = DictState(memory=Undeclared(tokenizer=encoding))
    for _ in range(5):
        state = copy_state_for_edit(state)

    assert RegistryEncoding.rebuilds > 0


def test_tiktoken_backed_tokenizer_copies_stay_cheap() -> None:
    """Same regression against the real thing, when tiktoken is installed."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # no network, no cached BPE
        pytest.skip(f"tiktoken encoding unavailable: {exc}")

    tokenizer = enc.encode
    state = DictState(memory=Memoryish(messages=["hi"], tokenizer=tokenizer))

    started = time.perf_counter()
    for _ in range(8):
        state = copy_state_for_edit(state)
    elapsed = time.perf_counter() - started

    assert state["memory"].tokenizer is tokenizer
    # A detached Encoding rebuilds in ~80 ms, so the unfixed path needs >500 ms
    # for these copies. The bound is loose enough for a loaded CI machine.
    assert elapsed < 0.25, f"copies degraded: {elapsed:.3f}s"
