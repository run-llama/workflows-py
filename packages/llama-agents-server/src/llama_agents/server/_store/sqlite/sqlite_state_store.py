# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import functools
import json
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Generic, Literal, Type

from pydantic import BaseModel
from typing_extensions import TypeVar
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.context.state_store import (
    DictState,
    create_cleared_state,
    deserialize_dict_state_data,
    deserialize_state_from_dict,
    get_by_path,
    merge_state,
    parse_in_memory_state,
    serialize_dict_state_data,
    set_by_path,
)

logger = logging.getLogger(__name__)

MODEL_T = TypeVar("MODEL_T", bound=BaseModel, default=DictState)  # type: ignore[reportGeneralTypeIssues]


class SqliteSerializedState(BaseModel):
    """Serialized state referencing a sqlite database row."""

    store_type: Literal["sqlite"] = "sqlite"
    run_id: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SqliteStateStore(Generic[MODEL_T]):
    """Sqlite-backed StateStore implementation.

    Every get() reads from the database, every set() writes through.
    No in-memory cache — the database is the source of truth.
    """

    state_type: Type[MODEL_T]

    def __init__(
        self,
        db_path: str,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> None:
        self._db_path = db_path
        self._run_id = run_id
        self.state_type = state_type or DictState  # type: ignore[assignment]
        self._serializer = serializer or JsonSerializer()

    @property
    def run_id(self) -> str:
        return self._run_id

    @functools.cached_property
    def _lock(self) -> asyncio.Lock:
        """Lazy lock initialization for Python 3.14+ compatibility."""
        return asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _write_in_memory_state(self, serialized_state: dict[str, Any]) -> None:
        """Migrate InMemory-format state into the database."""
        state = deserialize_state_from_dict(serialized_state, self._serializer)
        self._save_state(state)  # type: ignore[arg-type]

    def _serialize_state(self, state: MODEL_T) -> str:
        """Serialize state model to JSON string."""
        if isinstance(state, DictState):
            return json.dumps(serialize_dict_state_data(state, self._serializer))
        return self._serializer.serialize(state)

    def _deserialize_state(self, state_json: str) -> MODEL_T:
        """Deserialize state from JSON string."""
        if issubclass(self.state_type, DictState):
            data = json.loads(state_json)
            return deserialize_dict_state_data(data, self._serializer)  # type: ignore[return-value]
        return self._serializer.deserialize(state_json)

    def _create_default_state(self) -> MODEL_T:
        return self.state_type()

    def _load_state(self) -> MODEL_T:
        """Load state from database. Creates default if row doesn't exist."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT state_json FROM state WHERE run_id = ?",
                (self._run_id,),
            )
            row = cursor.fetchone()
            if row is None:
                state = self._create_default_state()
                self._save_state(state, conn)
                conn.commit()
                return state
            return self._deserialize_state(row[0])
        finally:
            conn.close()

    def _save_state(
        self, state: MODEL_T, conn: sqlite3.Connection | None = None
    ) -> None:
        """Save state to database."""
        should_close = conn is None
        if conn is None:
            conn = self._connect()
        try:
            now = _utc_now().isoformat()
            state_json = self._serialize_state(state)
            conn.execute(
                """
                INSERT INTO state (run_id, state_json, state_type, state_module, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    state_type = excluded.state_type,
                    state_module = excluded.state_module,
                    updated_at = excluded.updated_at
                """,
                (
                    self._run_id,
                    state_json,
                    type(state).__name__,
                    type(state).__module__,
                    now,
                    now,
                ),
            )
            if should_close:
                conn.commit()
        finally:
            if should_close:
                conn.close()

    async def get_state(self) -> MODEL_T:
        """Return a copy of the current state model."""
        state = self._load_state()
        return state.model_copy()

    async def set_state(self, state: MODEL_T) -> None:
        """Replace or merge into the current state model."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT state_json FROM state WHERE run_id = ?",
                (self._run_id,),
            )
            row = cursor.fetchone()

            if row is None:
                self._save_state(state, conn)
                conn.commit()
                return

            current_state = self._deserialize_state(row[0])
            merged = merge_state(current_state, state)
            self._save_state(merged, conn)  # type: ignore[arg-type]
            conn.commit()
        finally:
            conn.close()

    async def get(self, path: str, default: Any = ...) -> Any:
        """Get a nested value using dot-separated paths."""
        state = self._load_state()
        return get_by_path(state, path, default)

    async def set(self, path: str, value: Any) -> None:
        """Set a nested value using dot-separated paths."""
        async with self.edit_state() as state:
            set_by_path(state, path, value)

    async def clear(self) -> None:
        """Reset the state to its type defaults."""
        await self.set_state(create_cleared_state(self.state_type))

    @asynccontextmanager
    async def edit_state(self) -> AsyncGenerator[MODEL_T, None]:
        """Edit state transactionally under a lock."""
        async with self._lock:
            state = self._load_state()
            yield state
            self._save_state(state)

    def to_dict(self, serializer: BaseSerializer) -> dict[str, Any]:
        """Serialize state store metadata for persistence.

        Returns metadata only — actual state lives in the database.
        """
        payload = SqliteSerializedState(run_id=self._run_id)
        return payload.model_dump()

    @classmethod
    def from_dict(
        cls,
        serialized_state: dict[str, Any],
        serializer: BaseSerializer,
        db_path: str | None = None,
        state_type: type[BaseModel] | None = None,
        run_id: str | None = None,
    ) -> SqliteStateStore[Any]:
        """Restore a state store from serialized payload.

        Handles both InMemorySerializedState (migrates data to DB on first use)
        and SqliteSerializedState (reconnects to existing row).
        """
        if not serialized_state:
            raise ValueError("Cannot restore SqliteStateStore from empty dict")

        store_type = serialized_state.get("store_type")

        if store_type == "sqlite":
            parsed = SqliteSerializedState.model_validate(serialized_state)
            effective_run_id = run_id or parsed.run_id
            if db_path is None:
                raise ValueError("db_path is required for SqliteStateStore.from_dict()")
            return cls(
                db_path=db_path,
                run_id=effective_run_id,
                state_type=state_type,  # type: ignore[arg-type]
                serializer=serializer,
            )

        # InMemory format — migrate data to DB immediately
        parse_in_memory_state(serialized_state)

        effective_run_id = run_id or str(uuid.uuid4())
        if db_path is None:
            raise ValueError("db_path is required for SqliteStateStore.from_dict()")
        store = cls(
            db_path=db_path,
            run_id=effective_run_id,
            state_type=state_type,  # type: ignore[arg-type]
            serializer=serializer,
        )
        store._write_in_memory_state(serialized_state)
        return store
