# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""
SQL-backed StateStore implementations for durable workflow state.

Provides PostgreSQL and SQLite state stores that persist workflow state
to a database, enabling durable and distributed workflow execution.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generic,
    Literal,
    Type,
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import (
    Column,
    Connection,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from typing_extensions import TypeVar
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.context.state_store import (
    MAX_DEPTH,
    DictState,
    InMemorySerializedState,
    assign_path_step,
    deserialize_state_from_dict,
    traverse_path_step,
)

logger = logging.getLogger(__name__)

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)


class SqlSerializedState(BaseModel):
    """Serialized state referencing a database row (from SqlStateStore)."""

    model_config = ConfigDict(populate_by_name=True)

    store_type: Literal["sql"] = "sql"
    run_id: str
    db_schema: str | None = Field(default=None, alias="schema")
    # Note: No state_data - actual data lives in the database


def parse_serialized_state(
    data: dict[str, Any],
) -> InMemorySerializedState | SqlSerializedState:
    """Parse raw dict into appropriate format type.

    Args:
        data: Serialized state payload from to_dict().

    Returns:
        InMemorySerializedState or SqlSerializedState based on store_type.

    Raises:
        ValueError: If store_type is unknown.
    """
    store_type = data.get("store_type")

    if store_type == "sql":
        return SqlSerializedState.model_validate(data)
    elif store_type == "in_memory" or store_type is None:
        # Backwards compat: missing store_type = InMemory
        return InMemorySerializedState.model_validate(data)
    else:
        raise ValueError(f"Unknown store_type: {store_type}")


def _utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


STATE_TABLE_NAME = "workflow_state"


def _state_columns() -> list[Column]:
    """Return fresh Column instances for the workflow_state table.

    Must return new instances each call because SQLAlchemy Column objects
    can't be shared across Table instances.
    """
    return [
        Column("run_id", String(255), primary_key=True),
        Column("state_json", Text, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    ]


class SqlStateStore(Generic[MODEL_T]):
    """
    SQL-backed StateStore implementation.

    Persists workflow state to a database table. Supports PostgreSQL and SQLite
    dialects with automatic detection based on the engine.

    Thread-safety is achieved through database-level locking during
    transactional edits via the `edit_state` context manager.
    """

    known_unserializable_keys = ("memory",)
    state_type: Type[MODEL_T]

    def __init__(
        self,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        engine: Engine | None = None,
        serializer: BaseSerializer | None = None,
        schema: str | None = None,
        table_name: str = STATE_TABLE_NAME,
    ) -> None:
        self._run_id = run_id
        self.state_type = state_type or DictState  # type: ignore[assignment]
        self._engine = engine
        self._serializer = serializer or JsonSerializer()
        self._schema = schema
        self._table_name = table_name
        self._metadata = MetaData(schema=self._schema)
        self._table = self._define_table()
        self._initialized = False
        self._pending_state: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        """Get the workflow run ID."""
        return self._run_id

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine, raising if not set."""
        if self._engine is None:
            raise RuntimeError(
                "Engine not set. Provide an engine at construction or set it before use."
            )
        return self._engine

    @engine.setter
    def engine(self, engine: Engine) -> None:
        """Set the SQLAlchemy engine."""
        self._engine = engine

    @property
    def _is_postgres(self) -> bool:
        """Check if the engine is PostgreSQL."""
        return self.engine.dialect.name == "postgresql"

    @property
    def _table_ref(self) -> str:
        """Get the fully qualified table reference."""
        if self._schema:
            return f"{self._schema}.{self._table_name}"
        return self._table_name

    def _define_table(self) -> Table:
        """Define the workflow_state table schema."""
        return Table(
            self._table_name,
            self._metadata,
            *_state_columns(),
        )

    @classmethod
    def run_migrations(
        cls,
        engine: Engine,
        table_name: str = STATE_TABLE_NAME,
        schema: str | None = None,
    ) -> None:
        """Create schema and table if they don't exist."""
        metadata = MetaData(schema=schema)
        table = Table(
            table_name,
            metadata,
            *_state_columns(),
        )
        is_postgres = engine.dialect.name == "postgresql"
        with engine.begin() as conn:
            if is_postgres and schema:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))  # noqa: S608
            table.create(bind=conn, checkfirst=True)

    @functools.cached_property
    def _lock(self) -> asyncio.Lock:
        """Lazy lock for Python 3.14+ compatibility."""
        return asyncio.Lock()

    async def _run_sync(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a synchronous function in the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    def _ensure_initialized(self) -> None:
        """Ensure the table exists and apply any pending state."""
        if self._initialized:
            return
        self._run_instance_migrations()
        self._initialized = True

        # Apply any pending state from InMemory format deserialization
        if self._pending_state is not None:
            self._apply_pending_state_sync()

    def _apply_pending_state_sync(self) -> None:
        """Write pending state to database (called after migrations)."""
        if self._pending_state is None:
            return

        serialized_state = self._pending_state
        self._pending_state = None

        state = deserialize_state_from_dict(
            serialized_state, self._serializer, state_type=self.state_type
        )
        state_json = self._serialize_state(state)  # type: ignore[arg-type]

        with self.engine.begin() as conn:
            self._upsert_state(conn, state_json, _utc_now())

    def _run_instance_migrations(self) -> None:
        """Create schema and table if they don't exist (instance-level)."""
        with self.engine.begin() as conn:
            if self._is_postgres and self._schema:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self._schema}"))  # noqa: S608
            self._table.create(bind=conn, checkfirst=True)

    def _lock_row_for_update(self, conn: Connection) -> dict[str, Any] | None:
        """Lock and return row data for this run_id."""
        for_update = "FOR UPDATE" if self._is_postgres else ""
        result = conn.execute(
            text(f"""
                SELECT state_json
                FROM {self._table_ref}
                WHERE run_id = :run_id
                {for_update}
            """),  # noqa: S608
            {"run_id": self._run_id},
        )
        row = result.fetchone()
        if row is None:
            return None
        return {"state_json": row[0]}

    def _upsert_state(
        self,
        conn: Connection,
        state_json: str,
        now: datetime,
    ) -> None:
        """Perform database-specific upsert operation."""
        if self._is_postgres:
            stmt = pg_insert(self._table).values(
                run_id=self._run_id,
                state_json=state_json,
                created_at=now,
                updated_at=now,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["run_id"],
                set_={
                    "state_json": stmt.excluded.state_json,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            conn.execute(stmt)
        else:
            # SQLite upsert
            conn.execute(
                text(f"""
                    INSERT INTO {self._table_ref}
                        (run_id, state_json, created_at, updated_at)
                    VALUES (:run_id, :state_json, :created_at, :updated_at)
                    ON CONFLICT (run_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at
                """),  # noqa: S608
                {
                    "run_id": self._run_id,
                    "state_json": state_json,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                },
            )

    def _serialize_state(self, state: MODEL_T) -> str:
        """Serialize state model to JSON string."""
        if isinstance(state, DictState):
            serialized_data: dict[str, Any] = {}
            for key, value in state.items():
                try:
                    serialized_data[key] = self._serializer.serialize(value)
                except Exception:
                    if key in self.known_unserializable_keys:
                        logger.warning(f"Skipping unserializable key: {key}")
                        continue
                    raise
            return json.dumps({"_data": serialized_data})
        return self._serializer.serialize(state)

    def _deserialize_state(self, state_json: str) -> MODEL_T:
        """Deserialize state from JSON string."""
        if issubclass(self.state_type, DictState):
            data = json.loads(state_json)
            deserialized = {
                k: self._serializer.deserialize(v)
                for k, v in data.get("_data", {}).items()
            }
            return DictState(_data=deserialized)  # type: ignore[return-value]
        return self._serializer.deserialize(state_json)

    def _create_default_state(self) -> MODEL_T:
        """Create a default instance of the state type."""
        return self.state_type()

    def _load_state_sync(self) -> MODEL_T:
        """Load state from database synchronously."""
        self._ensure_initialized()
        with self.engine.connect() as conn:
            result = conn.execute(
                select(self._table.c.state_json).where(
                    self._table.c.run_id == self._run_id
                )
            )
            row = result.fetchone()
            if row is None:
                state = self._create_default_state()
                self._save_state_sync(state, conn)
                conn.commit()
                return state
            return self._deserialize_state(row[0])

    def _save_state_sync(self, state: MODEL_T, conn: Connection) -> None:
        """Save state to database synchronously."""
        now = _utc_now()
        self._upsert_state(conn, self._serialize_state(state), now)

    async def get_state(self) -> MODEL_T:
        """Return a copy of the current state model."""
        state = await self._run_sync(self._load_state_sync)
        return state.model_copy()

    async def set_state(self, state: MODEL_T) -> None:
        """Replace or merge into the current state model."""

        def _set_state_sync() -> None:
            self._ensure_initialized()
            with self.engine.begin() as conn:
                result = conn.execute(
                    select(self._table.c.state_json).where(
                        self._table.c.run_id == self._run_id
                    )
                )
                row = result.fetchone()

                if row is None:
                    self._save_state_sync(state, conn)
                    return

                current_state = self._deserialize_state(row[0])
                current_type = type(current_state)
                new_type = type(state)

                if isinstance(state, current_type):
                    self._save_state_sync(state, conn)
                elif issubclass(current_type, new_type):
                    parent_data = state.model_dump()
                    merged = current_type.model_validate(
                        {**current_state.model_dump(), **parent_data}
                    )
                    self._save_state_sync(merged, conn)
                else:
                    raise ValueError(
                        f"State must be of type {current_type.__name__} or parent, "
                        f"got {new_type.__name__}"
                    )

        await self._run_sync(_set_state_sync)

    async def get(self, path: str, default: Any = ...) -> Any:
        """Get a nested value using dot-separated paths."""
        state = await self._run_sync(self._load_state_sync)
        segments = path.split(".") if path else []
        if len(segments) > MAX_DEPTH:
            raise ValueError(f"Path length exceeds {MAX_DEPTH} segments")

        try:
            value: Any = state
            for segment in segments:
                value = traverse_path_step(value, segment)
        except Exception:
            if default is not ...:
                return default
            raise ValueError(f"Path '{path}' not found in state")
        return value

    async def set(self, path: str, value: Any) -> None:
        """Set a nested value using dot-separated paths."""
        if not path:
            raise ValueError("Path cannot be empty")
        segments = path.split(".")
        if len(segments) > MAX_DEPTH:
            raise ValueError(f"Path length exceeds {MAX_DEPTH} segments")

        async with self.edit_state() as state:
            current: Any = state
            for segment in segments[:-1]:
                try:
                    current = traverse_path_step(current, segment)
                except (KeyError, AttributeError, IndexError, TypeError):
                    intermediate: Any = {}
                    assign_path_step(current, segment, intermediate)
                    current = intermediate
            assign_path_step(current, segments[-1], value)

    async def clear(self) -> None:
        """Reset the state to its type defaults."""
        try:
            await self.set_state(self._create_default_state())
        except ValidationError:
            raise ValueError("State must have defaults for all fields")

    @asynccontextmanager
    async def edit_state(self) -> AsyncGenerator[MODEL_T, None]:
        """Edit state transactionally under a database lock."""

        def _edit_with_lock() -> tuple[
            MODEL_T, Callable[[MODEL_T], None], Callable[[], None]
        ]:
            self._ensure_initialized()
            conn = self.engine.connect()
            trans = conn.begin()
            finalized = False

            try:
                row_data = self._lock_row_for_update(conn)
                if row_data is None:
                    state = self._create_default_state()
                else:
                    state = self._deserialize_state(row_data["state_json"])

                def commit_fn(updated_state: MODEL_T) -> None:
                    nonlocal finalized
                    if finalized:
                        return
                    try:
                        self._save_state_sync(updated_state, conn)
                        trans.commit()
                    finally:
                        finalized = True
                        conn.close()

                def rollback_fn() -> None:
                    nonlocal finalized
                    if finalized:
                        return
                    try:
                        if trans.is_active:
                            trans.rollback()
                    finally:
                        finalized = True
                        conn.close()

                return state, commit_fn, rollback_fn
            except Exception:
                trans.rollback()
                conn.close()
                raise

        async with self._lock:
            state, commit_fn, rollback_fn = await self._run_sync(_edit_with_lock)
            try:
                yield state
                await self._run_sync(commit_fn, state)
            except Exception:
                try:
                    await self._run_sync(rollback_fn)
                except Exception:
                    logger.exception("Failed to rollback edit_state transaction")
                raise

    def to_dict(self, serializer: BaseSerializer) -> dict[str, Any]:
        """Serialize state store metadata for persistence.

        Returns a SqlSerializedState payload that can be restored by from_dict().
        The actual state data lives in the database, so this only serializes
        connection metadata (run_id, schema).
        """
        payload = SqlSerializedState.model_validate(
            {
                "run_id": self._run_id,
                "schema": self._schema,
            }
        )
        return payload.model_dump(by_alias=True)

    @classmethod
    def from_dict(
        cls,
        serialized_state: dict[str, Any],
        serializer: BaseSerializer,
        state_type: type[BaseModel] = DictState,
        run_id: str | None = None,
    ) -> SqlStateStore[Any]:
        """Restore a state store from serialized payload.

        Handles both InMemorySerializedState and SqlSerializedState formats:

        - InMemorySerializedState: Stores the serialized data internally and
          writes it to the database when the engine is first used (via
          _ensure_initialized). This enables restoring state from in-memory
          format into a SQL-backed store.

        - SqlSerializedState: Creates a store pointing at the existing database
          row. If a different run_id is provided, the store will use that run_id
          (data copying must be handled separately if needed).

        Note: The engine must be set separately after restoration.

        Args:
            serialized_state: Payload from to_dict() of either store type.
            serializer: Serializer for data handling.
            state_type: The state model type for deserialization.
            run_id: Optional override run_id. If not provided, uses the run_id
                from SqlSerializedState or generates one for InMemory format.

        Returns:
            A new SqlStateStore instance configured from the payload.

        Raises:
            ValueError: If serialized_state is empty.
        """
        import uuid

        if not serialized_state:
            raise ValueError("Cannot restore SqlStateStore from empty dict")

        parsed = parse_serialized_state(serialized_state)

        if isinstance(parsed, InMemorySerializedState):
            # InMemory format: store data internally, apply when engine is set
            effective_run_id = run_id or str(uuid.uuid4())

            store = cls(
                run_id=effective_run_id,
                state_type=state_type,  # type: ignore[arg-type]
                serializer=serializer,
            )
            # Store the serialized state to apply when engine is available
            store._pending_state = serialized_state
            return store

        else:
            # SqlSerializedState format: create store pointing at existing row
            effective_run_id = run_id or parsed.run_id
            schema = parsed.db_schema

            return cls(
                run_id=effective_run_id,
                state_type=state_type,  # type: ignore[arg-type]
                serializer=serializer,
                schema=schema,
            )
