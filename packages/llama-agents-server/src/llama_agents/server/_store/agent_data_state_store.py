# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""AgentDataStateStore — StateStore backed by the LlamaCloud Agent Data API."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Generic, Literal, Type

import httpx
from pydantic import BaseModel
from typing_extensions import TypeVar
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.context.state_store import (
    DictState,
    create_cleared_state,
    deserialize_dict_state_data,
    get_by_path,
    merge_state,
    serialize_dict_state_data,
    set_by_path,
)

logger = logging.getLogger(__name__)

MODEL_T = TypeVar("MODEL_T", bound=BaseModel, default=DictState)  # type: ignore[reportGeneralTypeIssues]


class AgentDataSerializedState(BaseModel):
    """Serialized state referencing an agent data store."""

    store_type: Literal["agent_data"] = "agent_data"
    run_id: str


class AgentDataStateStore(Generic[MODEL_T]):
    """StateStore backed by the LlamaCloud Agent Data API.

    Uses a single item in a ``workflow_state`` collection, keyed by ``run_id``.
    """

    state_type: Type[MODEL_T]

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        project_id: str,
        deployment_name: str,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        collection: str = "workflow_state",
        serializer: BaseSerializer | None = None,
        client_factory: Any | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._project_id = project_id
        self._deployment_name = deployment_name
        self._run_id = run_id
        self.state_type = state_type or DictState  # type: ignore[assignment]
        self._collection = collection
        self._serializer = serializer or JsonSerializer()
        self._client_factory = client_factory
        # Cache the agent data item ID once found
        self._item_id: str | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @functools.cached_property
    def _lock(self) -> asyncio.Lock:
        return asyncio.Lock()

    # ------------------------------------------------------------------
    # HTTP helpers (same pattern as AgentDataStore)
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers(),
            params={"project_id": self._project_id},
        )

    async def _search(
        self, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "deployment_name": self._deployment_name,
            "collection": self._collection,
            "page_size": 10,
        }
        if filters:
            body["filter"] = filters
        async with self._client() as client:
            resp = await client.post("/api/v1/beta/agent-data/:search", json=body)
            resp.raise_for_status()
            return resp.json().get("items", [])

    async def _create(self, data: dict[str, Any]) -> dict[str, Any]:
        body = {
            "deployment_name": self._deployment_name,
            "collection": self._collection,
            "data": data,
        }
        async with self._client() as client:
            resp = await client.post("/api/v1/beta/agent-data", json=body)
            resp.raise_for_status()
            return resp.json()

    async def _update_item(self, item_id: str, data: dict[str, Any]) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.put(
                f"/api/v1/beta/agent-data/{item_id}",
                json={"data": data},
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def _serialize_state(self, state: MODEL_T) -> str:
        if isinstance(state, DictState):
            return json.dumps(serialize_dict_state_data(state, self._serializer))
        return self._serializer.serialize(state)

    def _deserialize_state(self, state_json: str) -> MODEL_T:
        if issubclass(self.state_type, DictState):
            data = json.loads(state_json)
            return deserialize_dict_state_data(data, self._serializer)  # type: ignore[return-value]
        return self._serializer.deserialize(state_json)

    def _create_default_state(self) -> MODEL_T:
        return self.state_type()

    # ------------------------------------------------------------------
    # Load / save through API
    # ------------------------------------------------------------------

    async def _load_state(self) -> MODEL_T:
        items = await self._search({"run_id": {"eq": self._run_id}})
        if items:
            self._item_id = items[0]["id"]
            return self._deserialize_state(items[0]["data"]["state_json"])
        state = self._create_default_state()
        await self._save_state(state)
        return state

    async def _load_state_or_none(self) -> MODEL_T | None:
        items = await self._search({"run_id": {"eq": self._run_id}})
        if items:
            self._item_id = items[0]["id"]
            return self._deserialize_state(items[0]["data"]["state_json"])
        return None

    async def _save_state(self, state: MODEL_T) -> None:
        state_json = self._serialize_state(state)
        data = {
            "run_id": self._run_id,
            "state_json": state_json,
            "state_type": type(state).__name__,
            "state_module": type(state).__module__,
        }
        if self._item_id is not None:
            await self._update_item(self._item_id, data)
        else:
            items = await self._search({"run_id": {"eq": self._run_id}})
            if items:
                item_id = items[0]["id"]
                self._item_id = item_id
                await self._update_item(item_id, data)
            else:
                result = await self._create(data)
                self._item_id = result["id"]

    # ------------------------------------------------------------------
    # StateStore protocol
    # ------------------------------------------------------------------

    async def get_state(self) -> MODEL_T:
        state = await self._load_state()
        return state.model_copy()

    async def set_state(self, state: MODEL_T) -> None:
        current = await self._load_state_or_none()
        if current is None:
            await self._save_state(state)
            return
        merged = merge_state(current, state)
        await self._save_state(merged)  # type: ignore[arg-type]

    async def get(self, path: str, default: Any = ...) -> Any:
        state = await self._load_state()
        return get_by_path(state, path, default)

    async def set(self, path: str, value: Any) -> None:
        async with self.edit_state() as state:
            set_by_path(state, path, value)

    async def clear(self) -> None:
        await self._save_state(create_cleared_state(self.state_type))

    @asynccontextmanager
    async def edit_state(self) -> AsyncGenerator[MODEL_T, None]:
        async with self._lock:
            state = await self._load_state()
            yield state
            await self._save_state(state)

    def to_dict(self, serializer: BaseSerializer) -> dict[str, Any]:
        payload = AgentDataSerializedState(run_id=self._run_id)
        return payload.model_dump()

    @classmethod
    def from_dict(
        cls,
        serialized_state: dict[str, Any],
        serializer: BaseSerializer,
        *,
        base_url: str,
        api_key: str,
        project_id: str,
        deployment_name: str,
        state_type: type[BaseModel] | None = None,
        run_id: str | None = None,
    ) -> AgentDataStateStore[Any]:
        if not serialized_state:
            raise ValueError("Cannot restore AgentDataStateStore from empty dict")
        parsed = AgentDataSerializedState.model_validate(serialized_state)
        effective_run_id = run_id or parsed.run_id
        return cls(
            base_url=base_url,
            api_key=api_key,
            project_id=project_id,
            deployment_name=deployment_name,
            run_id=effective_run_id,
            state_type=state_type,  # type: ignore[arg-type]
            serializer=serializer,
        )
