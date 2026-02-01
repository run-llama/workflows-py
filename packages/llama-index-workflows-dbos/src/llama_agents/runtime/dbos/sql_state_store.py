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
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generic,
    Type,
)

from pydantic import BaseModel, ValidationError
from sqlalchemy import (
    Column,
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
    assign_path_step,
    traverse_path_step,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)


def _utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _get_qualified_name(cls: type) -> str:
    """Get fully qualified module.class name for a type."""
    return f"{cls.__module__}.{cls.__name__}"


def _import_class(qualified_name: str) -> type:
    """Import a class from its fully qualified name."""
    module_path, class_name = qualified_name.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class SqlStateStore(Generic[MODEL_T]):
    """
    SQL-backed StateStore implementation.

    Persists workflow state to a database table. Supports PostgreSQL and SQLite
    dialects with automatic detection based on the engine.

    Thread-safety is achieved through database-level locking during
    transactional edits via the `edit_state` context manager.

    Examples:
        ```python
        from sqlalchemy import create_engine
        from llama_agents.runtime.dbos.sql_state_store import SqlStateStore

        # PostgreSQL
        engine = create_engine("postgresql://user:pass@host/db")
        store = SqlStateStore(run_id="my-workflow", engine=engine)

        # SQLite
        engine = create_engine("sqlite:///workflow.db")
        store = SqlStateStore(run_id="my-workflow", engine=engine)

        await store.set("counter", 0)
        value = await store.get("counter")
        ```
    """

    TABLE_NAME = "workflow_state"
    known_unserializable_keys = ("memory",)
    state_type: Type[MODEL_T]

    def __init__(
        self,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        engine: Engine | None = None,
        serializer: BaseSerializer | None = None,
        schema: str | None = None,
    ) -> None:
        """Initialize the SQL state store.

        Args:
            run_id: Unique identifier for this workflow run.
            state_type: Pydantic model type for state. Defaults to DictState.
            engine: SQLAlchemy engine. If None, must be set before use.
            serializer: Serializer for state values. Defaults to JsonSerializer.
            schema: Database schema name (PostgreSQL only). Defaults to "dbos".
        """
        self._run_id = run_id
        self.state_type = state_type or DictState  # type: ignore[assignment]
        self._engine = engine
        self._serializer = serializer or JsonSerializer()
        self._schema = schema
        self._metadata = MetaData(schema=schema)
        self._table = self._define_table()
        self._initialized = False

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
            return f"{self._schema}.{self.TABLE_NAME}"
        return self.TABLE_NAME

    def _define_table(self) -> Table:
        """Define the workflow_state table schema."""
        return Table(
            self.TABLE_NAME,
            self._metadata,
            Column("run_id", String(255), primary_key=True),
            Column("state_json", Text, nullable=False),
            Column("state_type", String(255), nullable=False),
            Column("state_module", String(255), nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )

    @functools.cached_property
    def _lock(self) -> asyncio.Lock:
        """Lazy lock for Python 3.14+ compatibility."""
        return asyncio.Lock()

    async def _run_sync(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a synchronous function in the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    def _ensure_initialized(self) -> None:
        """Ensure the table exists (run migrations if needed)."""
        if self._initialized:
            return
        self._run_migrations()
        self._initialized = True

    def _run_migrations(self) -> None:
        """Create schema and table if they don't exist."""
        with self.engine.begin() as conn:
            if self._is_postgres and self._schema:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self._schema}"))  # noqa: S608
            self._table.create(bind=conn, checkfirst=True)

    def _lock_row_for_update(self, conn: Connection) -> dict[str, Any] | None:
        """Lock and return row data for this run_id."""
        for_update = "FOR UPDATE" if self._is_postgres else ""
        result = conn.execute(
            text(f"""
                SELECT state_json, state_type, state_module
                FROM {self._table_ref}
                WHERE run_id = :run_id
                {for_update}
            """),  # noqa: S608
            {"run_id": self._run_id},
        )
        row = result.fetchone()
        if row is None:
            return None
        return {"state_json": row[0], "state_type": row[1], "state_module": row[2]}

    def _upsert_state(
        self,
        conn: Connection,
        state_json: str,
        state_type_name: str,
        state_module: str,
        now: datetime,
    ) -> None:
        """Perform database-specific upsert operation."""
        if self._is_postgres:
            stmt = pg_insert(self._table).values(
                run_id=self._run_id,
                state_json=state_json,
                state_type=state_type_name,
                state_module=state_module,
                created_at=now,
                updated_at=now,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["run_id"],
                set_={
                    "state_json": stmt.excluded.state_json,
                    "state_type": stmt.excluded.state_type,
                    "state_module": stmt.excluded.state_module,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            conn.execute(stmt)
        else:
            # SQLite upsert
            conn.execute(
                text(f"""
                    INSERT INTO {self._table_ref}
                        (run_id, state_json, state_type, state_module, created_at, updated_at)
                    VALUES (:run_id, :state_json, :state_type, :state_module, :created_at, :updated_at)
                    ON CONFLICT (run_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        state_type = excluded.state_type,
                        state_module = excluded.state_module,
                        updated_at = excluded.updated_at
                """),  # noqa: S608
                {
                    "run_id": self._run_id,
                    "state_json": state_json,
                    "state_type": state_type_name,
                    "state_module": state_module,
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

    def _deserialize_state(
        self, state_json: str, state_type_name: str, state_module: str
    ) -> MODEL_T:
        """Deserialize state from JSON string."""
        if state_type_name == "DictState":
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
                select(
                    self._table.c.state_json,
                    self._table.c.state_type,
                    self._table.c.state_module,
                ).where(self._table.c.run_id == self._run_id)
            )
            row = result.fetchone()
            if row is None:
                state = self._create_default_state()
                self._save_state_sync(state, conn)
                conn.commit()
                return state
            return self._deserialize_state(row[0], row[1], row[2])

    def _save_state_sync(self, state: MODEL_T, conn: Connection) -> None:
        """Save state to database synchronously."""
        now = _utc_now()
        self._upsert_state(
            conn,
            self._serialize_state(state),
            type(state).__name__,
            type(state).__module__,
            now,
        )

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
                    select(
                        self._table.c.state_json,
                        self._table.c.state_type,
                        self._table.c.state_module,
                    ).where(self._table.c.run_id == self._run_id)
                )
                row = result.fetchone()

                if row is None:
                    self._save_state_sync(state, conn)
                    return

                current_state = self._deserialize_state(row[0], row[1], row[2])
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

        def _edit_with_lock() -> tuple[MODEL_T, Callable[[MODEL_T], None]]:
            self._ensure_initialized()
            conn = self.engine.connect()
            trans = conn.begin()

            try:
                row_data = self._lock_row_for_update(conn)
                if row_data is None:
                    state = self._create_default_state()
                else:
                    state = self._deserialize_state(
                        row_data["state_json"],
                        row_data["state_type"],
                        row_data["state_module"],
                    )

                def commit_fn(updated_state: MODEL_T) -> None:
                    try:
                        self._save_state_sync(updated_state, conn)
                        trans.commit()
                    finally:
                        conn.close()

                return state, commit_fn
            except Exception:
                trans.rollback()
                conn.close()
                raise

        async with self._lock:
            state, commit_fn = await self._run_sync(_edit_with_lock)
            try:
                yield state
                await self._run_sync(commit_fn, state)
            except Exception:
                raise

    async def load_raw(self, key: str) -> str | None:
        """Load raw JSON string by key.

        This is used for journal storage, separate from the main state model.

        Args:
            key: The key to load (used as run_id in the table)

        Returns:
            The raw JSON string if found, None otherwise
        """

        def _load() -> str | None:
            self._ensure_initialized()
            with self.engine.connect() as conn:
                result = conn.execute(
                    select(self._table.c.state_json).where(self._table.c.run_id == key)
                )
                row = result.fetchone()
                return row[0] if row else None

        return await self._run_sync(_load)

    async def save_raw(self, key: str, data: str) -> None:
        """Save raw JSON string by key (upsert).

        This is used for journal storage, separate from the main state model.

        Args:
            key: The key to save (used as run_id in the table)
            data: The raw JSON string to save
        """

        def _save_with_conn() -> None:
            self._ensure_initialized()
            now = _utc_now()
            with self.engine.begin() as conn:
                # Inline upsert to avoid swapping run_id
                if self._is_postgres:
                    stmt = pg_insert(self._table).values(
                        run_id=key,
                        state_json=data,
                        state_type="journal",
                        state_module="builtin",
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
                                (run_id, state_json, state_type, state_module, created_at, updated_at)
                            VALUES (:run_id, :state_json, :state_type, :state_module, :created_at, :updated_at)
                            ON CONFLICT (run_id) DO UPDATE SET
                                state_json = excluded.state_json,
                                updated_at = excluded.updated_at
                        """),  # noqa: S608
                        {
                            "run_id": key,
                            "state_json": data,
                            "state_type": "journal",
                            "state_module": "builtin",
                            "created_at": now.isoformat(),
                            "updated_at": now.isoformat(),
                        },
                    )

        await self._run_sync(_save_with_conn)

    def to_dict(self, serializer: BaseSerializer) -> dict[str, Any]:
        """Serialize state store metadata for persistence."""
        return {
            "run_id": self._run_id,
            "state_type": self.state_type.__name__,
            "state_module": self.state_type.__module__,
            "schema": self._schema,
            "store_class": _get_qualified_name(type(self)),
        }

    @classmethod
    def from_dict(
        cls,
        serialized_state: dict[str, Any],
        serializer: BaseSerializer,
    ) -> SqlStateStore[Any]:
        """Restore a state store from serialized metadata.

        Note: The engine must be set separately after restoration.
        """
        if not serialized_state:
            raise ValueError("Cannot restore SqlStateStore from empty dict")

        run_id = serialized_state["run_id"]
        state_type_name = serialized_state.get("state_type", "DictState")
        state_module = serialized_state.get(
            "state_module", "workflows.context.state_store"
        )
        schema = serialized_state.get("schema")

        if state_type_name == "DictState":
            state_type: type[BaseModel] = DictState
        else:
            state_type = _import_class(f"{state_module}.{state_type_name}")

        return cls(
            run_id=run_id,
            state_type=state_type,  # type: ignore[arg-type]
            serializer=serializer,
            schema=schema,
        )


# Convenience aliases for explicit dialect selection
class PostgresStateStore(SqlStateStore[MODEL_T]):
    """PostgreSQL-backed StateStore with 'dbos' schema default."""

    DEFAULT_SCHEMA = "dbos"

    def __init__(
        self,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        engine: Engine | None = None,
        serializer: BaseSerializer | None = None,
        schema: str | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            state_type=state_type,
            engine=engine,
            serializer=serializer,
            schema=schema or self.DEFAULT_SCHEMA,
        )


class SqliteStateStore(SqlStateStore[MODEL_T]):
    """SQLite-backed StateStore (no schema support)."""

    def __init__(
        self,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        engine: Engine | None = None,
        serializer: BaseSerializer | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            state_type=state_type,
            engine=engine,
            serializer=serializer,
            schema=None,
        )


def create_state_store(
    run_id: str,
    engine: Engine,
    state_type: type[BaseModel] | None = None,
    serializer: BaseSerializer | None = None,
) -> SqlStateStore[Any]:
    """Factory to create the appropriate state store for an engine.

    Automatically selects PostgresStateStore or SqliteStateStore based on dialect.
    """
    dialect = engine.dialect.name
    if dialect == "postgresql":
        return PostgresStateStore(
            run_id=run_id,
            state_type=state_type,  # type: ignore[arg-type]
            engine=engine,
            serializer=serializer,
        )
    elif dialect == "sqlite":
        return SqliteStateStore(
            run_id=run_id,
            state_type=state_type,  # type: ignore[arg-type]
            engine=engine,
            serializer=serializer,
        )
    else:
        raise ValueError(f"Unsupported database dialect: {dialect}")


__all__ = [
    "SqlStateStore",
    "PostgresStateStore",
    "SqliteStateStore",
    "create_state_store",
]
