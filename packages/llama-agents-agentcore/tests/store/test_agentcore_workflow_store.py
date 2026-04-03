import functools
from datetime import datetime
from typing import Any, cast
from unittest.mock import patch

import pytest
from bedrock_agentcore.memory.models.filters import StringValue
from llama_agents.agentcore.store.agentcore_state_store import (
    STATE_STORE_MEMORY_NAME,
    create_agentcore_payload,
)
from llama_agents.agentcore.store.agentcore_workflow_store import (
    DEFAULT_INTEGRATION_NAME,
    EVENTS_MEMORY_NAME,
    HANDLERS_ACTOR_ID,
    HANDLERS_MEMORY_NAME,
    HANDLERS_SESSION_ID,
    TICKS_MEMORY_NAME,
    AgentCoreWorkflowStore,
    _b64_str_to_model,
)
from llama_agents.client.protocol.serializable_events import EventEnvelopeWithMetadata
from llama_agents.server._store.abstract_workflow_store import (
    HandlerQuery,
    PersistentHandler,
    Status,
    StoredEvent,
    StoredTick,
)
from pydantic import BaseModel
from workflows.context import JsonSerializer
from workflows.context.serializers import PickleSerializer
from workflows.context.state_store import DictState
from workflows.events import Event, StartEvent, StopEvent

from .conftest import MockMemoryClient, make_data_event

_MODULE_NAME = "llama_agents.agentcore.store.agentcore_workflow_store.MemoryClient"

patch_deco = patch(_MODULE_NAME, new=MockMemoryClient)


def _make_persistent_handler(
    handler_id: str = "handler-1",
    workflow_name: str = "workflow",
    status: Status = "running",
    run_id: str | None = None,
    idle_since: datetime | None = None,
) -> PersistentHandler:
    return PersistentHandler(
        handler_id=handler_id,
        status=status,
        workflow_name=workflow_name,
        run_id=run_id,
        idle_since=idle_since,
    )


@functools.lru_cache(maxsize=1)
def _make_default_persistent_handlers() -> list[PersistentHandler]:
    workflow_names = ["workflow-1", "workflow-2"]
    run_ids = ["run-id-1", "run-id-2"]
    statuses: list[Status] = ["completed", "running"]
    idle_since = [datetime.now(), None]
    handlers = []
    for i in range(10):
        idx = i % 2
        handlers.append(
            _make_persistent_handler(
                handler_id=f"handler-{i}",
                run_id=run_ids[idx],
                status=statuses[idx],
                idle_since=idle_since[idx],
                workflow_name=workflow_names[idx],
            )
        )
    return handlers


class InputEvent(StartEvent):
    start: str = "start"


class SomeEvent(Event):
    data: str = "some_event"


class EndEvent(StopEvent):
    end: str = "end"


def _make_event(event: Event = SomeEvent()) -> EventEnvelopeWithMetadata:
    return EventEnvelopeWithMetadata.from_event(event)


def _make_default_events() -> dict[str, list[EventEnvelopeWithMetadata]]:
    events: dict[str, list[EventEnvelopeWithMetadata]] = {
        "run-id-1": [],
        "run-id-2": [],
        "run-id-3": [],
    }
    for key in events:
        start = EventEnvelopeWithMetadata.from_event(InputEvent())
        some = EventEnvelopeWithMetadata.from_event(SomeEvent())
        end = EventEnvelopeWithMetadata.from_event(EndEvent())
        events[key] = [start, some, end]
    return events


class TickData(BaseModel):
    name: str = "tick"
    input: str = "hello"
    output: str = "hello back"


def _make_tick_data() -> dict[str, Any]:
    return TickData().model_dump()


def _make_many_tick_data() -> dict[str, list[dict[str, Any]]]:
    ticks: dict[str, list[dict[str, Any]]] = {
        "run-id-1": [],
        "run-id-2": [],
        "run-id-3": [],
    }
    for key in ticks:
        ticks[key] = [TickData().model_dump()] * 3
    return ticks


@patch_deco
def test_init() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    assert store._client.region_name == "us-west-2"
    assert store._client.integration_source == DEFAULT_INTEGRATION_NAME


@patch_deco
@pytest.mark.asyncio
async def test_memory_failures() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    client = cast(MockMemoryClient, store._client)
    client.memories_failed = True
    with pytest.raises(RuntimeError, match="Failed to provision memory for"):
        await store._create_memories()


@patch_deco
@pytest.mark.asyncio
async def test_update() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    await store.update(
        handler=_make_persistent_handler(),
    )
    client = cast(MockMemoryClient, store._client)
    assert len(client.created_memories) == 3
    for memory in (HANDLERS_MEMORY_NAME, EVENTS_MEMORY_NAME, TICKS_MEMORY_NAME):
        assert memory in client.created_memories
    assert len(client.events) == 1
    assert (
        client.events[0]["eventId"]
        == f"{store.handlers_memory_id}-{HANDLERS_ACTOR_ID}-{HANDLERS_SESSION_ID}-{client.events_count}"
    )
    meta = client.events[0]["metadata"]
    assert meta is not None
    assert meta.get("handler_id") == StringValue(stringValue="handler-1")
    assert meta.get("workflow_name") == StringValue(stringValue="workflow")
    assert meta.get("run_id") == StringValue(
        stringValue=""
    )  # none values are converted to empty strings
    assert meta.get("is_idle") == StringValue(stringValue="no")
    assert meta.get("status") == StringValue(stringValue="running")


@patch_deco
@pytest.mark.asyncio
async def test_update_many() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    handlers = _make_default_persistent_handlers()
    for handler in handlers:
        await store.update(handler)
    client = cast(MockMemoryClient, store._client)
    assert len(client.created_memories) == 3
    for memory in (HANDLERS_MEMORY_NAME, EVENTS_MEMORY_NAME, TICKS_MEMORY_NAME):
        assert memory in client.created_memories
    assert len(client.events) == 10
    for i, event in enumerate(client.events):
        assert (
            event["eventId"]
            == f"{store.handlers_memory_id}-{HANDLERS_ACTOR_ID}-{HANDLERS_SESSION_ID}-{i + 1}"
        )
        meta = event["metadata"]
        assert meta is not None
        assert meta.get("handler_id") == StringValue(stringValue=f"handler-{i}")
        if i % 2 == 1:
            assert meta.get("run_id") == StringValue(stringValue="run-id-2")
            assert meta.get("is_idle") == StringValue(stringValue="no")
            assert meta.get("status") == StringValue(stringValue="running")
            assert meta.get("workflow_name") == StringValue(stringValue="workflow-2")
        else:
            assert meta.get("run_id") == StringValue(stringValue="run-id-1")
            assert meta.get("is_idle") == StringValue(stringValue="yes")
            assert meta.get("status") == StringValue(stringValue="completed")
            assert meta.get("workflow_name") == StringValue(stringValue="workflow-1")


@patch_deco
@pytest.mark.asyncio
async def test_query_no_filters() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    handlers = _make_default_persistent_handlers()
    for handler in handlers:
        await store.update(handler)
    query = HandlerQuery()
    results = await store.query(query)
    assert len(results) == len(handlers)
    for i, result in enumerate(results):
        assert result.handler_id == handlers[i].handler_id
        assert result.run_id == handlers[i].run_id
        assert result.workflow_name == handlers[i].workflow_name
        assert result.status == handlers[i].status
        assert result.idle_since == handlers[i].idle_since


@patch_deco
@pytest.mark.asyncio
async def test_query_handler_ids() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    handlers = _make_default_persistent_handlers()
    for handler in handlers:
        await store.update(handler)
    query = HandlerQuery(handler_id_in=["handler-1", "handler-2"])
    handlers_filt = [
        handler
        for handler in handlers
        if handler.handler_id in cast(list, query.handler_id_in)
    ]
    results = await store.query(query)
    assert len(results) == len(handlers_filt)
    for i, result in enumerate(handlers_filt):
        assert result.handler_id == handlers_filt[i].handler_id
        assert result.run_id == handlers_filt[i].run_id
        assert result.workflow_name == handlers_filt[i].workflow_name
        assert result.status == handlers_filt[i].status
        assert result.idle_since == handlers_filt[i].idle_since


@patch_deco
@pytest.mark.asyncio
async def test_query_workflow_name() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    handlers = _make_default_persistent_handlers()
    for handler in handlers:
        await store.update(handler)
    query = HandlerQuery(workflow_name_in=["workflow-1"])
    handlers_filt = [
        handler
        for handler in handlers
        if handler.workflow_name in cast(list, query.workflow_name_in)
    ]
    results = await store.query(query)
    assert len(results) == len(handlers_filt)
    for i, result in enumerate(handlers_filt):
        assert result.handler_id == handlers_filt[i].handler_id
        assert result.run_id == handlers_filt[i].run_id
        assert result.workflow_name == handlers_filt[i].workflow_name
        assert result.status == handlers_filt[i].status
        assert result.idle_since == handlers_filt[i].idle_since


@patch_deco
@pytest.mark.asyncio
async def test_query_status() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    handlers = _make_default_persistent_handlers()
    for handler in handlers:
        await store.update(handler)
    query = HandlerQuery(status_in=["running"])
    handlers_filt = [
        handler for handler in handlers if handler.status in cast(list, query.status_in)
    ]
    results = await store.query(query)
    assert len(results) == len(handlers_filt)
    for i, result in enumerate(handlers_filt):
        assert result.handler_id == handlers_filt[i].handler_id
        assert result.run_id == handlers_filt[i].run_id
        assert result.workflow_name == handlers_filt[i].workflow_name
        assert result.status == handlers_filt[i].status
        assert result.idle_since == handlers_filt[i].idle_since


@patch_deco
@pytest.mark.asyncio
async def test_query_idle() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    handlers = _make_default_persistent_handlers()
    for handler in handlers:
        await store.update(handler)
    query = HandlerQuery(is_idle=True)
    handlers_filt = [handler for handler in handlers if handler.idle_since is not None]
    results = await store.query(query)
    assert len(results) == len(handlers_filt)
    for i, result in enumerate(handlers_filt):
        assert result.handler_id == handlers_filt[i].handler_id
        assert result.run_id == handlers_filt[i].run_id
        assert result.workflow_name == handlers_filt[i].workflow_name
        assert result.status == handlers_filt[i].status
        assert result.idle_since == handlers_filt[i].idle_since


@patch_deco
@pytest.mark.asyncio
async def test_query_run_id() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    handlers = _make_default_persistent_handlers()
    for handler in handlers:
        await store.update(handler)
    query = HandlerQuery(run_id_in=["run-id-1"])
    handlers_filt = [
        handler for handler in handlers if handler.run_id in cast(list, query.run_id_in)
    ]
    results = await store.query(query)
    assert len(results) == len(handlers_filt)
    for i, result in enumerate(handlers_filt):
        assert result.handler_id == handlers_filt[i].handler_id
        assert result.run_id == handlers_filt[i].run_id
        assert result.workflow_name == handlers_filt[i].workflow_name
        assert result.status == handlers_filt[i].status
        assert result.idle_since == handlers_filt[i].idle_since


@patch_deco
@pytest.mark.asyncio
async def test_delete_one() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    await store.update(_make_persistent_handler())
    result = await store.query(HandlerQuery())
    assert len(result) == 1
    await store.delete(HandlerQuery(handler_id_in=["handler-1"]))
    client = cast(MockMemoryClient, store._client)
    assert len(client.gmdp_client.deleted_events) == 1
    assert (
        client.gmdp_client.deleted_events[0]
        == f"{store.handlers_memory_id}-{HANDLERS_ACTOR_ID}-{HANDLERS_SESSION_ID}-{client.events_count}"
    )
    result = await store.query(HandlerQuery())
    assert len(result) == 0


@patch_deco
@pytest.mark.asyncio
async def test_delete_many() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    handlers = _make_default_persistent_handlers()
    for handler in handlers:
        await store.update(handler)
    result = await store.query(HandlerQuery())
    assert len(result) == len(handlers)
    handlers_filt = [handler for handler in handlers if handler.idle_since is not None]
    handlers_other = [handler for handler in handlers if handler.idle_since is None]
    await store.delete(HandlerQuery(is_idle=True))
    client = cast(MockMemoryClient, store._client)
    assert len(client.gmdp_client.deleted_events) == len(handlers_other)
    sorted_events = sorted(
        client.gmdp_client.deleted_events, key=lambda s: int(s.split("-")[-1])
    )  # needs to be sorted because we are deleting concurrently, so the order of the deleted events is scrambled
    for i, event in enumerate(sorted_events):
        event_id_num = handlers_other[i].handler_id[-1]
        assert (
            event
            == f"{store.handlers_memory_id}-{HANDLERS_ACTOR_ID}-{HANDLERS_SESSION_ID}-{event_id_num}"
        )
    result = await store.query(HandlerQuery())
    assert len(result) == len(handlers_filt)


@patch_deco
@pytest.mark.asyncio
async def test_append_event() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    event_env = _make_event()
    run_id = "run_id"
    await store.append_event(run_id, event_env)
    client = cast(MockMemoryClient, store._client)
    assert len(client.created_memories) == 3
    for memory in (HANDLERS_MEMORY_NAME, EVENTS_MEMORY_NAME, TICKS_MEMORY_NAME):
        assert memory in client.created_memories
    assert len(store._events_tracker) == 1
    assert "run_id" in store._events_tracker
    assert len(store._events_tracker["run_id"]) == 1
    await store._regroup_events("run_id")
    assert len(client.events) == 1
    event = client.events[0]
    assert (
        event["eventId"]
        == f"{store.events_memory_id}-{run_id}-{run_id}-{client.events_count}"
    )
    payload = event["payload"]
    assert len(payload) == 1
    blob = payload[0]["blob"]
    model = _b64_str_to_model(blob)
    assert isinstance(model, StoredEvent)
    assert model.event.type == event_env.type
    assert model.event.value == event_env.value
    assert model.event.types == event_env.types
    assert model.event.qualified_name == event_env.qualified_name
    assert model.sequence == client.events_count
    assert model.run_id == run_id


@patch_deco
@pytest.mark.asyncio
async def test_query_events() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    events = _make_default_events()
    for key in events:
        for event in events[key]:
            await store.append_event(key, event)
    filt_key = next(iter(events))
    results = await store.query_events(run_id=filt_key)
    assert len(results) == 3
    filt_events = events[filt_key]
    for i, ev in enumerate(results):
        assert ev.event.model_dump() == filt_events[i].model_dump()


@patch_deco
@pytest.mark.asyncio
async def test_query_events_limit() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    events = _make_default_events()
    for key in events:
        for event in events[key]:
            await store.append_event(key, event)
    filt_key = next(iter(events))
    results = await store.query_events(run_id=filt_key, limit=2)
    assert len(results) == 2
    filt_events = events[filt_key]
    for i, ev in enumerate(results):
        assert ev.event.model_dump() == filt_events[i].model_dump()


@patch_deco
@pytest.mark.asyncio
async def test_query_events_after_sequence() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    events = _make_default_events()
    for key in events:
        for event in events[key]:
            await store.append_event(key, event)
    filt_key = next(iter(events))
    results = await store.query_events(
        run_id=filt_key, after_sequence=2
    )  # sequences for this events are 1,2,3, so only the last one (3) is retained
    assert len(results) == 1
    filt_events = events[filt_key]
    assert results[0].event.model_dump() == filt_events[2].model_dump()


@patch_deco
@pytest.mark.asyncio
async def test_append_tick() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    tick_data = _make_tick_data()
    run_id = "run_id"
    await store.append_tick(run_id, tick_data)
    client = cast(MockMemoryClient, store._client)
    assert len(client.created_memories) == 3
    for memory in (HANDLERS_MEMORY_NAME, EVENTS_MEMORY_NAME, TICKS_MEMORY_NAME):
        assert memory in client.created_memories
    assert len(store._ticks_tracker) == 1
    assert "run_id" in store._ticks_tracker
    assert len(store._ticks_tracker["run_id"]) == 1
    await store._regroup_ticks("run_id")
    assert len(client.events) == 1
    event = client.events[0]
    assert (
        event["eventId"]
        == f"{store.ticks_memory_id}-{run_id}-{run_id}-{client.events_count}"
    )
    payload = event["payload"]
    assert len(payload) == 1
    blob = payload[0]["blob"]
    model = _b64_str_to_model(blob)
    assert isinstance(model, StoredTick)
    assert model.tick_data == tick_data
    assert model.run_id == run_id
    assert model.sequence == client.events_count


@patch_deco
@pytest.mark.asyncio
async def test_get_ticks() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    tick_data_d = _make_many_tick_data()
    for run_id in tick_data_d:
        for tick_data in tick_data_d[run_id]:
            await store.append_tick(run_id, tick_data)
    run_id = next(iter(tick_data_d))
    ticks = await store.get_ticks(
        run_id=run_id
    )  # these are the first 3 ticks, so sequences are 1,2,3
    assert len(ticks) == 3
    filt_ticks = tick_data_d[run_id]
    for i, tick in enumerate(ticks):
        assert filt_ticks[i] == tick.tick_data
        assert tick.run_id == run_id
        assert tick.sequence == i + 1


@patch_deco
def test_create_state_store_default() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    state_store = store.create_state_store(run_id="run_id")
    assert isinstance(state_store.serializer, JsonSerializer)
    assert state_store.state_type is DictState
    assert state_store.run_id == "run_id"
    assert state_store._in_memory_state is not None


@patch_deco
def test_create_state_store_custom() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    state_store = store.create_state_store(
        run_id="run_id",
        state_type=TickData,
        serializer=PickleSerializer(),
    )
    assert state_store.state_type is TickData
    assert state_store.run_id == "run_id"
    assert isinstance(state_store.serializer, PickleSerializer)
    assert state_store._in_memory_state is not None
    assert state_store._in_memory_state.name == "tick"
    assert state_store._in_memory_state.input == "hello"
    assert state_store._in_memory_state.output == "hello back"


@patch_deco
def test_create_state_store_custom_state_from_serialized() -> None:
    store = AgentCoreWorkflowStore(region_name="us-west-2")
    client = cast(MockMemoryClient, store._client)
    event_id, event = make_data_event(
        TickData(name="hello", input="world", output="hello world"), JsonSerializer()
    )
    client.gmdp_client._append_event(event)
    payload = create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    state_store = store.create_state_store(
        run_id="run_id",
        state_type=TickData,
        serialized_state=payload,
    )
    assert state_store.state_type is TickData
    assert state_store.run_id == "run_id"
    assert state_store._in_memory_state is not None
    assert state_store._in_memory_state.name == "hello"
    assert state_store._in_memory_state.input == "world"
    assert state_store._in_memory_state.output == "hello world"
