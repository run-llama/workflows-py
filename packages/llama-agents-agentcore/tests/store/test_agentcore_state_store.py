import pytest
from llama_agents.agentcore.store.agentcore_state_store import (
    STATE_STORE_ACTOR_ID,
    STATE_STORE_MEMORY_NAME,
    AgentCoreStateStore,
    create_agentcore_payload,
)
from pydantic import BaseModel
from workflows.context.serializers import (
    JsonSerializer,
    PickleSerializer,
)
from workflows.context.state_store import DictState, create_in_memory_payload

from .conftest import MockMemoryClient, make_data_event, serialize_state

_MODULE_NAME = "llama_agents.agentcore.store.agentcore_workflow_store.MemoryClient"


class CustomState(BaseModel):
    key: str = "key"
    value: str = "value"


class OtherCustomState(BaseModel):
    key: int = 1
    value: str = "value"


def test_state_store_init_defaults() -> None:
    state_store = AgentCoreStateStore(
        client=MockMemoryClient(
            region_name="us-west-2",
            integration_source="test-integration",
        ),  # type: ignore
        run_id="run-id",
    )
    assert state_store.state_type is DictState
    assert isinstance(state_store.serializer, JsonSerializer)
    assert state_store.run_id == "run-id"
    assert state_store._in_memory_state is None
    assert state_store._current_state_snapshot is None
    assert state_store._client.integration_source == "test-integration"
    assert state_store._client.region_name == "us-west-2"


def test_state_store_init_custom() -> None:
    state_store = AgentCoreStateStore(
        client=MockMemoryClient(
            region_name="us-west-2",
            integration_source="test-integration",
        ),  # type: ignore
        run_id="run-id",
        state_type=CustomState,
        serializer=PickleSerializer(),
    )
    assert state_store.state_type is CustomState
    assert isinstance(state_store.serializer, PickleSerializer)
    assert state_store.run_id == "run-id"
    assert state_store._in_memory_state is None
    assert state_store._current_state_snapshot is None
    assert state_store._client.integration_source == "test-integration"
    assert state_store._client.region_name == "us-west-2"


def test_state_store_init_state() -> None:
    state_store = AgentCoreStateStore(
        client=MockMemoryClient(
            region_name="us-west-2",
            integration_source="test-integration",
        ),  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._init_state()
    assert state_store._in_memory_state is not None
    assert isinstance(state_store._in_memory_state, CustomState)
    assert state_store._in_memory_state.key == "key"
    assert state_store._in_memory_state.value == "value"


def test_state_store_seed_from_serialized_success() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(CustomState(key="k", value="v"), JsonSerializer())
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._seed_from_serialized(
        create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    )
    assert state_store._in_memory_state is not None
    assert isinstance(state_store._in_memory_state, CustomState)
    assert state_store._in_memory_state.key == "k"
    assert state_store._in_memory_state.value == "v"


def test_state_store_seed_from_serialized_validation_error() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(OtherCustomState(), JsonSerializer())
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(TypeError, match="State is not of type CustomState"):
        state_store._seed_from_serialized(
            create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
        )


def test_state_store_seed_from_serialized_value_error() -> None:
    state_store = AgentCoreStateStore(
        client=MockMemoryClient(
            region_name="us-west-2",
            integration_source="test-integration",
        ),  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(ValueError, match="Unsupported state type: 'in_memory'"):
        payload = create_in_memory_payload(CustomState(), JsonSerializer()).model_dump()
        state_store._seed_from_serialized(payload)


def test_state_store_seed_from_serialized_key_error() -> None:
    state_store = AgentCoreStateStore(
        client=MockMemoryClient(
            region_name="us-west-2",
            integration_source="test-integration",
        ),  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(KeyError, match="Missing required key"):
        payload = create_agentcore_payload(
            "hello", STATE_STORE_MEMORY_NAME
        ).model_dump()
        payload.pop("state_snapshot_id")
        state_store._seed_from_serialized(
            payload,
        )


def test_state_store_seed_from_serialized_other_key_error() -> None:
    state_store = AgentCoreStateStore(
        client=MockMemoryClient(
            region_name="us-west-2",
            integration_source="test-integration",
        ),  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(KeyError, match="Missing required key"):
        payload = create_agentcore_payload(
            "hello", STATE_STORE_MEMORY_NAME
        ).model_dump()
        payload.pop("state_memory_id")
        state_store._seed_from_serialized(
            payload,
        )


def test_state_store_seed_from_serialized_no_event_error() -> None:
    state_store = AgentCoreStateStore(
        client=MockMemoryClient(
            region_name="us-west-2",
            integration_source="test-integration",
        ),  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(
        ValueError, match="No event associated with state_snapshot_id hello"
    ):
        payload = create_agentcore_payload(
            "hello", STATE_STORE_MEMORY_NAME
        ).model_dump()
        state_store._seed_from_serialized(
            payload,
        )


def test_state_store_seed_from_serialized_no_data_error() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(
        CustomState(key="k", value="v"), JsonSerializer(), "without-data"
    )
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(
        ValueError, match="No data associated with state_snapshot_id without-data"
    ):
        payload = create_agentcore_payload(
            event_id, STATE_STORE_MEMORY_NAME
        ).model_dump()
        state_store._seed_from_serialized(
            payload,
        )


@pytest.mark.asyncio
async def test_state_store_memory_failure() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    mock_client.memories_failed = True
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(
        RuntimeError, match="Failed to provision memory for state store"
    ):
        await state_store._create_memory()


@pytest.mark.asyncio
async def test_state_store_set_state_from_blank_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    await state_store.set_state(CustomState(key="k"))
    assert state_store._in_memory_state is not None
    assert state_store._in_memory_state.key == "k"
    assert state_store._in_memory_state.value == "value"
    assert (
        state_store._current_state_snapshot
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert len(mock_client.events) == 1
    assert (
        mock_client.events[0]["eventId"]
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert mock_client.events[0]["metadata"] is None
    assert len(mock_client.events[0]["payload"]) == 1
    assert mock_client.events[0]["payload"][0]["blob"] == serialize_state(
        state_store.serializer, state_store._in_memory_state
    )
    assert len(mock_client.created_memories) == 1
    assert mock_client.created_memories[0] == STATE_STORE_MEMORY_NAME
    assert mock_client.checked_state_store


@pytest.mark.asyncio
async def test_state_store_set_state_merge_states() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(CustomState(key="k", value="v"), JsonSerializer())
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._seed_from_serialized(
        create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    )
    await state_store.set_state(CustomState(key="key1"))
    assert state_store._in_memory_state is not None
    assert state_store._in_memory_state.key == "key1"
    assert state_store._in_memory_state.value == "v"
    assert (
        state_store._current_state_snapshot
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert len(mock_client.events) == 1
    assert (
        mock_client.events[0]["eventId"]
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert mock_client.events[0]["metadata"] is None
    assert len(mock_client.events[0]["payload"]) == 1
    assert mock_client.events[0]["payload"][0]["blob"] == serialize_state(
        state_store.serializer, state_store._in_memory_state
    )
    assert len(mock_client.created_memories) == 1
    assert mock_client.created_memories[0] == STATE_STORE_MEMORY_NAME
    assert (
        not mock_client.checked_state_store
    )  # was seeded from an existing one, so state store has not been checked


@pytest.mark.asyncio
async def test_state_store_set_blank_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    await state_store.set("key", "test")
    assert state_store._in_memory_state is not None
    assert state_store._in_memory_state.key == "test"
    assert (
        state_store._in_memory_state.value == "value"
    )  # state was initialized from defaults
    assert (
        state_store._current_state_snapshot
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert len(mock_client.events) == 1
    assert (
        mock_client.events[0]["eventId"]
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert mock_client.events[0]["metadata"] is None
    assert len(mock_client.events[0]["payload"]) == 1
    assert mock_client.events[0]["payload"][0]["blob"] == serialize_state(
        state_store.serializer, state_store._in_memory_state
    )
    assert len(mock_client.created_memories) == 1
    assert mock_client.created_memories[0] == STATE_STORE_MEMORY_NAME
    assert mock_client.checked_state_store


@pytest.mark.asyncio
async def test_state_store_set_existing_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(CustomState(key="k", value="v"), JsonSerializer())
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._seed_from_serialized(
        create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    )
    await state_store.set("key", "test")
    assert state_store._in_memory_state is not None
    assert state_store._in_memory_state.key == "test"
    assert state_store._in_memory_state.value == "v"
    assert (
        state_store._current_state_snapshot
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert len(mock_client.events) == 1
    assert (
        mock_client.events[0]["eventId"]
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert mock_client.events[0]["metadata"] is None
    assert len(mock_client.events[0]["payload"]) == 1
    assert mock_client.events[0]["payload"][0]["blob"] == serialize_state(
        state_store.serializer, state_store._in_memory_state
    )
    assert len(mock_client.created_memories) == 1
    assert mock_client.created_memories[0] == STATE_STORE_MEMORY_NAME
    assert not mock_client.checked_state_store


@pytest.mark.asyncio
async def test_state_store_set_key_empty_path() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(ValueError, match="Path cannot be empty"):
        await state_store.set("", "test")
    assert mock_client.events_count == 0  # state has not been saved


@pytest.mark.asyncio
async def test_state_store_set_key_not_exist() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(
        ValueError, match='"CustomState" object has no field "notexist"'
    ):
        await state_store.set("notexist", "test")
    assert mock_client.events_count == 0  # state has not been saved


@pytest.mark.asyncio
async def test_state_store_get_state_blank_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state = await state_store.get_state()  # gets default state
    assert state.key == "key"
    assert state.value == "value"
    assert len(mock_client.gmdp_client.gotten_events) == 0  # accessed state from memory


@pytest.mark.asyncio
async def test_state_store_get_state_existing_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(CustomState(key="k", value="v"), JsonSerializer())
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._seed_from_serialized(
        create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    )
    state = await state_store.get_state()  # gets default state
    assert state.key == "k"
    assert state.value == "v"
    assert len(mock_client.gmdp_client.gotten_events) == 0  # accessed state from memory


@pytest.mark.asyncio
async def test_state_store_set_get_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    await state_store.set_state(CustomState(key="hello", value="world"))
    assert state_store._current_state_snapshot is not None
    state = await state_store.get_state()  # gets state stored in AgentCore
    assert state.key == "hello"
    assert state.value == "world"
    assert (
        len(mock_client.gmdp_client.gotten_events) == 0
    )  # everything is sourced from the in-memory state if it is not None


@pytest.mark.asyncio
async def test_state_store_get_blank_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    result = await state_store.get("key")
    assert result == "key"
    assert len(mock_client.gmdp_client.gotten_events) == 0  # accessed state from memory


@pytest.mark.asyncio
async def test_state_store_get_existing_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(
        CustomState(key="bye", value="moon"), JsonSerializer()
    )
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._seed_from_serialized(
        create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    )
    result = await state_store.get("value")
    assert result == "moon"
    assert len(mock_client.gmdp_client.gotten_events) == 0  # accessed state from memory


@pytest.mark.asyncio
async def test_state_store_get_default_fallback() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    result = await state_store.get("notexist")
    assert result is None
    result1 = await state_store.get("notexist", "some")
    assert result1 == "some"
    assert len(mock_client.gmdp_client.gotten_events) == 0  # accessed state from memory


@pytest.mark.asyncio
async def test_state_store_set_get() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    await state_store.set("value", "v")
    result = await state_store.get("value")
    assert result == "v"
    assert (
        len(mock_client.gmdp_client.gotten_events) == 0
    )  # everything is sourced from the in-memory state if it is not None


@pytest.mark.asyncio
async def test_state_store_clear() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(CustomState(key="k", value="v"), JsonSerializer())
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._seed_from_serialized(
        create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    )
    result = await state_store.get("key")
    assert result == "k"
    result1 = await state_store.get("value")
    assert result1 == "v"
    await state_store.clear()
    assert (
        state_store._current_state_snapshot
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert len(mock_client.events) == 1
    assert (
        mock_client.events[0]["eventId"]
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert mock_client.events[0]["metadata"] is None
    assert len(mock_client.events[0]["payload"]) == 1
    assert mock_client.events[0]["payload"][0]["blob"] == serialize_state(
        state_store.serializer, state_store._in_memory_state
    )
    assert len(mock_client.created_memories) == 1
    assert mock_client.created_memories[0] == STATE_STORE_MEMORY_NAME
    assert not mock_client.checked_state_store

    result2 = await state_store.get("key")
    assert result2 == "key"  # reset to default value
    result3 = await state_store.get("value")
    assert result3 == "value"  # reset to default value


@pytest.mark.asyncio
async def test_state_store_edit_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(CustomState(key="k", value="v"), JsonSerializer())
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._seed_from_serialized(
        create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    )
    result = await state_store.get("key")
    assert result == "k"
    result1 = await state_store.get("value")
    assert result1 == "v"

    async with state_store.edit_state() as state:
        state.key = "hello"
        state.value = "world"

    assert (
        state_store._current_state_snapshot
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert len(mock_client.events) == 1
    assert (
        mock_client.events[0]["eventId"]
        == f"{state_store.state_store_memory_id}-{STATE_STORE_ACTOR_ID}-{state_store.run_id}-{mock_client.events_count}"
    )
    assert mock_client.events[0]["metadata"] is None
    assert len(mock_client.events[0]["payload"]) == 1
    assert mock_client.events[0]["payload"][0]["blob"] == serialize_state(
        state_store.serializer, state_store._in_memory_state
    )
    assert len(mock_client.created_memories) == 1
    assert mock_client.created_memories[0] == STATE_STORE_MEMORY_NAME
    assert not mock_client.checked_state_store

    result2 = await state_store.get("key")
    assert result2 == "hello"  # edited value
    result3 = await state_store.get("value")
    assert result3 == "world"  # edited value


def test_state_store_to_dict_blank_state_error() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    with pytest.raises(
        ValueError,
        match="State cannot be dumped to dict as it has not been uploaded to AgentCore Memory yet",
    ):
        state_store.to_dict(serializer=JsonSerializer())


def test_state_store_to_dict_existing_state() -> None:
    mock_client = MockMemoryClient(
        region_name="us-west-2",
        integration_source="test-integration",
    )
    event_id, event = make_data_event(CustomState(key="k", value="v"), JsonSerializer())
    mock_client.gmdp_client._append_event(event)
    state_store = AgentCoreStateStore(
        client=mock_client,  # type: ignore
        run_id="run-id",
        state_type=CustomState,
    )
    state_store._seed_from_serialized(
        create_agentcore_payload(event_id, STATE_STORE_MEMORY_NAME).model_dump()
    )
    data = state_store.to_dict(serializer=JsonSerializer())
    assert data.get("store_type") == "agentcore"
    state_data = data.get("state_snapshot_id")
    assert state_data is not None
    assert state_data == event_id
