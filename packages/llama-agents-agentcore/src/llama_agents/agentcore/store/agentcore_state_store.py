import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Generic, Literal, Type, TypeVar, cast

from bedrock_agentcore.memory import MemoryClient
from pydantic import BaseModel
from workflows.context import BaseSerializer, JsonSerializer
from workflows.context.state_store import (
    DictState,
    create_cleared_state,
    get_by_path,
    set_by_path,
)

from ._utils import check_memory_activity, create_memory
from .errors import StateNotFoundError

STATE_STORE_ACTOR_ID = "state-store-actor-llamaindex"
STATE_STORE_MEMORY_NAME = "state_store_memory_llamaindex"

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)


class AgentCoreSerializedState(BaseModel):
    store_type: Literal["agentcore"] = "agentcore"
    state_snapshot_id: str
    state_memory_id: str


def create_agentcore_payload(
    state_snapshot_id: str,
    state_memory_id: str,
) -> AgentCoreSerializedState:
    """Create AgentCoreSerializedState from any state model.

    Args:
        state_snapshot_id: the ID of the event holding the latest state snapshot (as saved in AgentCore Memory)

    Returns:
        AgentCoreSerializedState containing the state_snapshot_id.
    """

    return AgentCoreSerializedState(
        state_snapshot_id=state_snapshot_id, state_memory_id=state_memory_id
    )


class AgentCoreStateStore(Generic[MODEL_T]):
    def __init__(
        self,
        client: MemoryClient,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> None:
        self.state_type: Type[MODEL_T] = state_type or DictState  # type: ignore
        self.serializer = serializer or JsonSerializer()
        self.run_id = run_id
        self._in_memory_state: MODEL_T | None = None
        self._client: MemoryClient = client
        self._current_state_snapshot: str | None = None
        self._memory_inited = False
        self._state_store_memory_id: str | None = None

    @property
    def _lock(self) -> asyncio.Lock:
        return asyncio.Lock()

    @property
    def state_store_memory_id(self) -> str:
        if self._state_store_memory_id is None:
            raise ValueError("Cannot access a None property as string")
        return self._state_store_memory_id

    def _init_state(self) -> None:
        if self._in_memory_state is None:
            self._in_memory_state = self.state_type()

    async def _locked_init_state(self) -> None:
        async with self._lock:
            self._init_state()

    # supposed to run only once per lifetime, on memory initialization, to check if there are already
    # events associated with the provided run_id
    async def _check_for_existing_state(self) -> None:
        if self._current_state_snapshot is None:
            response = await asyncio.to_thread(
                self._client.list_events,
                memory_id=self.state_store_memory_id,
                actor_id=STATE_STORE_ACTOR_ID,
                session_id=self.run_id,
            )
            if len(response) > 0:
                response.sort(
                    key=lambda x: x.get("eventTimestamp") or datetime.min, reverse=True
                )
                self._current_state_snapshot = response[0].get("eventId")

    async def _create_memory(self) -> None:
        if not self._memory_inited:
            memory_id = await create_memory(
                self._client, STATE_STORE_MEMORY_NAME, "state store"
            )
            self._state_store_memory_id = memory_id
            is_active = await check_memory_activity(
                self._client, self.state_store_memory_id
            )
            if not is_active:
                raise RuntimeError("Failed to provision memory for state store")
            self._memory_inited = True
            await self._check_for_existing_state()

    def _seed_from_serialized(self, serialized_state: dict[str, Any]) -> None:
        if serialized_state.get("store_type") == "agentcore":
            state_snapshot_id = serialized_state.get("state_snapshot_id")
            state_memory_id = serialized_state.get("state_memory_id")
            if state_snapshot_id is not None and state_memory_id is not None:
                response = self._client.gmdp_client.get_event(
                    eventId=state_snapshot_id,
                    memoryId=state_memory_id,
                    sessionId=self.run_id,
                    actorId=STATE_STORE_ACTOR_ID,
                )
                event = response.get("event")
                if event is not None:
                    payload = event.get("payload", [])
                    data = payload[0].get("blob") if len(payload) > 0 else None
                    if data is not None:
                        state = self.serializer.deserialize(
                            base64.b64decode(data).decode("utf-8")
                        )
                        if not isinstance(state, self.state_type):
                            raise TypeError(
                                f"State is not of type {self.state_type.__name__}"
                            )
                        self._in_memory_state = state
                        self._current_state_snapshot = state_snapshot_id
                        self._state_store_memory_id = state_memory_id
                    else:
                        raise ValueError(
                            f"No data associated with state_snapshot_id {state_snapshot_id}"
                        )
                else:
                    raise ValueError(
                        f"No event associated with state_snapshot_id {state_snapshot_id}"
                    )
            else:
                raise KeyError(
                    f"Missing required key(s): {'state_snapshot_id' if state_snapshot_id is None else ''} {'state_memory_id' if state_memory_id is None else ''}"
                )
        else:
            raise ValueError(
                f"Unsupported state type: '{serialized_state.get('store_type')}'"
            )

    async def _save_state(self) -> None:
        await self._create_memory()
        await self._locked_init_state()
        async with self._lock:
            serialized = self.serializer.serialize(self._in_memory_state).encode(
                "utf-8"
            )
        event = await asyncio.to_thread(
            self._client.create_blob_event,
            memory_id=self.state_store_memory_id,
            session_id=self.run_id,
            actor_id=STATE_STORE_ACTOR_ID,
            blob_data=base64.b64encode(serialized).decode("utf-8"),
        )
        if (event_id := event.get("eventId")) is not None:
            self._current_state_snapshot = event_id
        else:
            raise RuntimeError("Event creation did not return an event ID")

    async def _load_state(self) -> None:
        if (
            self._current_state_snapshot is not None and self._in_memory_state is None
        ):  # only load from remote state if no local state is available
            response = await asyncio.to_thread(
                self._client.gmdp_client.get_event,
                eventId=self._current_state_snapshot,
                memoryId=self.state_store_memory_id,
                sessionId=self.run_id,
                actorId=STATE_STORE_ACTOR_ID,
            )
            event = response.get("event")
            if event is not None:
                payload = event.get("payload", [])
                data = payload[0].get("blob") if len(payload) > 0 else None
                if data is not None:
                    decoded = base64.b64decode(data).decode("utf-8")
                    state = self.serializer.deserialize(decoded)
                    async with self._lock:
                        self._in_memory_state = state
                else:
                    raise StateNotFoundError(cause="data")
            else:
                raise StateNotFoundError(cause="event_id")
        elif self._current_state_snapshot is None and self._in_memory_state is None:
            await self._locked_init_state()

    async def get_state(self) -> MODEL_T:
        await self._load_state()
        return cast(MODEL_T, self._in_memory_state).model_copy()

    async def _set_state(self, state: MODEL_T, clear: bool = False):
        await self._load_state()
        async with self._lock:
            if clear:
                self._in_memory_state = state
            else:
                merged = cast(
                    MODEL_T, self._in_memory_state
                ).model_dump() | state.model_dump(exclude_unset=True)
                self._in_memory_state = self._in_memory_state.__class__(**merged)
        await self._save_state()

    async def set_state(self, state: MODEL_T) -> None:
        await self._set_state(state)

    async def get(self, path: str, default: Any = None) -> Any:
        await self._load_state()
        async with self._lock:
            return get_by_path(self._in_memory_state, path, default)

    async def set(self, path: str, value: Any) -> None:
        await self._load_state()
        async with self._lock:
            set_by_path(self._in_memory_state, path, value)
        await self._save_state()

    async def clear(self) -> None:
        await self._set_state(create_cleared_state(self.state_type), clear=True)

    @asynccontextmanager
    async def edit_state(self) -> AsyncGenerator[MODEL_T, None]:
        await self._load_state()
        state = cast(MODEL_T, self._in_memory_state)
        async with self._lock:
            yield state

            self._in_memory_state = state
        await self._save_state()

    def to_dict(self, serializer: BaseSerializer) -> dict[str, Any]:
        if self._current_state_snapshot is None:
            raise ValueError(
                "State cannot be dumped to dict as it has not been uploaded to AgentCore Memory yet"
            )
        payload = create_agentcore_payload(
            self._current_state_snapshot, self.state_store_memory_id
        )
        return payload.model_dump()
