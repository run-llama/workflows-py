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
from abc import ABC, abstractmethod
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
from workflows.context.state_store import DictState

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

# Default state type
MODEL_T = TypeVar("MODEL_T", bound=BaseModel)

# Type alias for executor functions
F = TypeVar("F", bound=Callable[..., Any])


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


class SqlStateStore(ABC, Generic[MODEL_T]):
    """
    Base class for SQL-backed StateStore implementations.

    Persists workflow state to a database table with the schema:
    - run_id: TEXT PRIMARY KEY
    - state_json: TEXT (serialized state data)
    - state_type: TEXT (class name)
    - state_module: TEXT (module path)
    - created_at: TIMESTAMP
    - updated_at: TIMESTAMP

    Subclasses must implement database-specific migration and locking behavior.

    Thread-safety is achieved through database-level locking during
    transactional edits via the `edit_state` context manager.

    Examples:
        Using PostgreSQL:

        ```python
        from sqlalchemy import create_engine
        from llama_agents.runtime.dbos.sql_state_store import PostgresStateStore

        engine = create_engine("postgresql://user:pass@host/db")
        store = PostgresStateStore(run_id="my-workflow", engine=engine)
        await store.set("counter", 0)
        value = await store.get("counter")
        ```

    See Also:
        - [InMemoryStateStore][workflows.context.state_store.InMemoryStateStore]
        - [StateStore][workflows.context.state_store.StateStore]
    """

    # Table name for workflow state
    TABLE_NAME = "workflow_state"

    # Keys known to be unserializable (mirrors InMemoryStateStore)
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
            schema: Database schema name. Defaults to None (use default schema).
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
        """Lazy lock initialization for Python 3.14+ compatibility.

        asyncio.Lock() requires a running event loop in Python 3.14+.
        Using cached_property defers creation to first use in async context.
        """
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

    @abstractmethod
    def _run_migrations(self) -> None:
        """Run database-specific migrations to create/update the table.

        Must be idempotent - should check if table exists before creating.
        """
        ...

    @abstractmethod
    def _lock_row_for_update(self, conn: Connection) -> dict[str, Any] | None:
        """Lock the row for this run_id and return its data.

        Uses database-specific locking:
        - PostgreSQL: SELECT ... FOR UPDATE
        - SQLite: Exclusive transaction

        Args:
            conn: Active database connection.

        Returns:
            Row data as dict if exists, None otherwise.
        """
        ...

    def _serialize_state(self, state: MODEL_T) -> str:
        """Serialize state model to JSON string."""
        if isinstance(state, DictState):
            # Special handling for DictState - serialize each item
            serialized_data: dict[str, Any] = {}
            for key, value in state.items():
                try:
                    serialized_data[key] = self._serializer.serialize(value)
                except Exception:
                    if key in self.known_unserializable_keys:
                        logger.warning(
                            f"Skipping serialization of unserializable key: {key}"
                        )
                        continue
                    raise
            return json.dumps({"_data": serialized_data})
        else:
            # Regular Pydantic model
            serialized = self._serializer.serialize(state)
            return serialized

    def _deserialize_state(
        self, state_json: str, state_type_name: str, state_module: str
    ) -> MODEL_T:
        """Deserialize state from JSON string."""
        if state_type_name == "DictState":
            data = json.loads(state_json)
            _data_serialized = data.get("_data", {})
            deserialized_data = {}
            for key, value in _data_serialized.items():
                deserialized_data[key] = self._serializer.deserialize(value)
            return DictState(_data=deserialized_data)  # type: ignore[return-value]
        else:
            return self._serializer.deserialize(state_json)

    def _create_default_state(self) -> MODEL_T:
        """Create a default instance of the state type."""
        return self.state_type()

    def _load_state_sync(self) -> MODEL_T:
        """Load state from database synchronously. Creates default if not exists."""
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
                # Create default state and persist it
                state = self._create_default_state()
                self._save_state_sync(state, conn)
                conn.commit()
                return state

            return self._deserialize_state(row[0], row[1], row[2])

    def _save_state_sync(self, state: MODEL_T, conn: Connection) -> None:
        """Save state to database synchronously within an existing connection."""
        now = _utc_now()
        state_json = self._serialize_state(state)
        state_type_name = type(state).__name__
        state_module = type(state).__module__

        # Use upsert (database-specific)
        self._upsert_state(conn, state_json, state_type_name, state_module, now)

    @abstractmethod
    def _upsert_state(
        self,
        conn: Connection,
        state_json: str,
        state_type_name: str,
        state_module: str,
        now: datetime,
    ) -> None:
        """Perform database-specific upsert operation."""
        ...

    async def get_state(self) -> MODEL_T:
        """Return a copy of the current state model.

        If no state exists in the database, creates and persists a default state.

        Returns:
            MODEL_T: A copy of the state model.
        """
        state = await self._run_sync(self._load_state_sync)
        return state.model_copy()

    async def set_state(self, state: MODEL_T) -> None:
        """Replace or merge into the current state model.

        If the provided state is the exact type of the current state, it replaces
        the state entirely. If the provided state is a parent type (i.e., the
        current state type is a subclass of the provided state type), the fields
        from the parent are merged onto the current state.

        Args:
            state: New state, either the same type or a parent type.

        Raises:
            ValueError: If the types are not compatible.
        """

        def _set_state_sync() -> None:
            self._ensure_initialized()
            with self.engine.begin() as conn:
                # Load current state to check type compatibility
                result = conn.execute(
                    select(
                        self._table.c.state_json,
                        self._table.c.state_type,
                        self._table.c.state_module,
                    ).where(self._table.c.run_id == self._run_id)
                )
                row = result.fetchone()

                if row is None:
                    # No existing state - just save
                    self._save_state_sync(state, conn)
                    return

                current_state = self._deserialize_state(row[0], row[1], row[2])
                current_type = type(current_state)
                new_type = type(state)

                if isinstance(state, current_type):
                    # Exact match or subclass - direct replacement
                    self._save_state_sync(state, conn)
                elif issubclass(current_type, new_type):
                    # Parent type provided - merge fields
                    parent_data = state.model_dump()
                    merged_state = current_type.model_validate(
                        {**current_state.model_dump(), **parent_data}
                    )
                    self._save_state_sync(merged_state, conn)
                else:
                    raise ValueError(
                        f"State must be of type {current_type.__name__} or a parent type, "
                        f"got {new_type.__name__}"
                    )

        await self._run_sync(_set_state_sync)

    async def get(self, path: str, default: Any = ...) -> Any:
        """Get a nested value using dot-separated paths.

        Supports dict keys, list indices, and attribute access transparently
        at each segment.

        Args:
            path: Dot-separated path, e.g. "user.profile.name".
            default: If provided, return this when the path does not exist;
                otherwise, raise ValueError.

        Returns:
            The resolved value.

        Raises:
            ValueError: If the path is invalid and no default is provided.
        """
        state = await self._run_sync(self._load_state_sync)

        segments = path.split(".") if path else []
        if len(segments) > 1000:
            raise ValueError("Path length exceeds 1000 segments")

        try:
            value: Any = state
            for segment in segments:
                value = self._traverse_step(value, segment)
        except Exception:
            if default is not ...:
                return default
            raise ValueError(f"Path '{path}' not found in state")

        return value

    async def set(self, path: str, value: Any) -> None:
        """Set a nested value using dot-separated paths.

        Intermediate containers are created as needed.

        Args:
            path: Dot-separated path to write.
            value: Value to assign.

        Raises:
            ValueError: If the path is empty or exceeds the maximum depth.
        """
        if not path:
            raise ValueError("Path cannot be empty")

        segments = path.split(".")
        if len(segments) > 1000:
            raise ValueError("Path length exceeds 1000 segments")

        async with self.edit_state() as state:
            current: Any = state

            # Navigate/create intermediate segments
            for segment in segments[:-1]:
                try:
                    current = self._traverse_step(current, segment)
                except (KeyError, AttributeError, IndexError, TypeError):
                    # Create intermediate object and assign it
                    intermediate: Any = {}
                    self._assign_step(current, segment, intermediate)
                    current = intermediate

            # Assign the final value
            self._assign_step(current, segments[-1], value)

    async def clear(self) -> None:
        """Reset the state to its type defaults.

        Raises:
            ValueError: If the model type cannot be instantiated from defaults.
        """
        try:
            default_state = self._create_default_state()
            await self.set_state(default_state)
        except ValidationError:
            raise ValueError("State must have defaults for all fields")

    @asynccontextmanager
    async def edit_state(self) -> AsyncGenerator[MODEL_T, None]:
        """Edit state transactionally under a database lock.

        Uses database-level locking to ensure atomic updates:
        - PostgreSQL: SELECT ... FOR UPDATE
        - SQLite: Exclusive transaction

        Yields:
            The current state model for in-place mutation.
        """

        def _edit_with_lock() -> tuple[MODEL_T, Callable[[MODEL_T], None]]:
            self._ensure_initialized()
            conn = self.engine.connect()
            trans = conn.begin()

            try:
                # Lock the row (database-specific)
                row_data = self._lock_row_for_update(conn)

                if row_data is None:
                    # Create default state
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

        # Use the async lock to serialize access at the Python level too
        async with self._lock:
            state, commit_fn = await self._run_sync(_edit_with_lock)

            try:
                yield state
                await self._run_sync(commit_fn, state)
            except Exception:
                # Rollback happens in the finally block of _edit_with_lock
                raise

    def to_dict(self, serializer: BaseSerializer) -> dict[str, Any]:
        """Serialize state store metadata for persistence.

        For SQL state stores, this returns connection metadata rather than
        the full state contents, since state lives in the database.

        Args:
            serializer: Serializer (used for consistency with protocol).

        Returns:
            A payload suitable for from_dict reconstruction.
        """
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

        Args:
            serialized_state: Payload from to_dict.
            serializer: Serializer for state values.

        Returns:
            A new SqlStateStore instance (engine must be set before use).
        """
        if not serialized_state:
            raise ValueError("Cannot restore SqlStateStore from empty dict")

        run_id = serialized_state["run_id"]
        state_type_name = serialized_state.get("state_type", "DictState")
        state_module = serialized_state.get(
            "state_module", "workflows.context.state_store"
        )
        schema = serialized_state.get("schema")

        # Import the state type
        if state_type_name == "DictState":
            state_type: type[BaseModel] = DictState
        else:
            qualified_name = f"{state_module}.{state_type_name}"
            state_type = _import_class(qualified_name)

        return cls(
            run_id=run_id,
            state_type=state_type,  # type: ignore[arg-type]
            serializer=serializer,
            schema=schema,
        )

    def _traverse_step(self, obj: Any, segment: str) -> Any:
        """Follow one segment into obj (dict key, list index, or attribute)."""
        if isinstance(obj, dict):
            return obj[segment]

        # Attempt list/tuple index
        try:
            idx = int(segment)
            return obj[idx]
        except (ValueError, TypeError, IndexError):
            pass

        # Fallback to attribute access
        return getattr(obj, segment)

    def _assign_step(self, obj: Any, segment: str, value: Any) -> None:
        """Assign value to segment of obj (dict key, list index, or attribute)."""
        if isinstance(obj, dict):
            obj[segment] = value
            return

        # Attempt list/tuple index assignment
        try:
            idx = int(segment)
            obj[idx] = value
            return
        except (ValueError, TypeError, IndexError):
            pass

        # Fallback to attribute assignment
        setattr(obj, segment, value)


class PostgresStateStore(SqlStateStore[MODEL_T]):
    """
    PostgreSQL-backed StateStore implementation.

    Uses the "dbos" schema by default (matching DBOS conventions).
    Uses SELECT ... FOR UPDATE for row-level locking during edits.

    Examples:
        ```python
        from sqlalchemy import create_engine
        from llama_agents.runtime.dbos.sql_state_store import PostgresStateStore

        engine = create_engine("postgresql://user:pass@host/db")
        store = PostgresStateStore(run_id="workflow-123", engine=engine)

        async with store.edit_state() as state:
            state.counter = state.get("counter", 0) + 1
        ```
    """

    DEFAULT_SCHEMA = "dbos"

    def __init__(
        self,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        engine: Engine | None = None,
        serializer: BaseSerializer | None = None,
        schema: str | None = None,
    ) -> None:
        """Initialize PostgreSQL state store.

        Args:
            run_id: Unique identifier for this workflow run.
            state_type: Pydantic model type for state. Defaults to DictState.
            engine: SQLAlchemy engine for PostgreSQL.
            serializer: Serializer for state values. Defaults to JsonSerializer.
            schema: Database schema name. Defaults to "dbos".
        """
        super().__init__(
            run_id=run_id,
            state_type=state_type,
            engine=engine,
            serializer=serializer,
            schema=schema or self.DEFAULT_SCHEMA,
        )

    def _run_migrations(self) -> None:
        """Create the dbos schema and workflow_state table if they don't exist."""
        with self.engine.begin() as conn:
            # Create schema if not exists
            conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")  # noqa: S608
            )

            # Check if table exists
            result = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = :schema
                        AND table_name = :table_name
                    )
                    """
                ),
                {"schema": self._schema, "table_name": self.TABLE_NAME},
            )
            exists = result.scalar()

            if not exists:
                # Create table
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE {self._schema}.{self.TABLE_NAME} (
                            run_id VARCHAR(255) PRIMARY KEY,
                            state_json TEXT NOT NULL,
                            state_type VARCHAR(255) NOT NULL,
                            state_module VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                        )
                        """  # noqa: S608
                    )
                )
                logger.info(
                    f"Created table {self._schema}.{self.TABLE_NAME} for workflow state"
                )

    def _lock_row_for_update(self, conn: Connection) -> dict[str, Any] | None:
        """Lock the row using SELECT ... FOR UPDATE."""
        result = conn.execute(
            text(
                f"""
                SELECT state_json, state_type, state_module
                FROM {self._schema}.{self.TABLE_NAME}
                WHERE run_id = :run_id
                FOR UPDATE
                """  # noqa: S608
            ),
            {"run_id": self._run_id},
        )
        row = result.fetchone()

        if row is None:
            return None

        return {
            "state_json": row[0],
            "state_type": row[1],
            "state_module": row[2],
        }

    def _upsert_state(
        self,
        conn: Connection,
        state_json: str,
        state_type_name: str,
        state_module: str,
        now: datetime,
    ) -> None:
        """Perform PostgreSQL upsert using ON CONFLICT DO UPDATE."""
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


class SqliteStateStore(SqlStateStore[MODEL_T]):
    """
    SQLite-backed StateStore implementation.

    Uses exclusive transactions for locking during edits.
    Suitable for local development and testing.

    Examples:
        ```python
        from sqlalchemy import create_engine
        from llama_agents.runtime.dbos.sql_state_store import SqliteStateStore

        engine = create_engine("sqlite:///workflow.db")
        store = SqliteStateStore(run_id="workflow-123", engine=engine)

        await store.set("user.name", "Alice")
        name = await store.get("user.name")
        ```

        In-memory database for testing:

        ```python
        engine = create_engine("sqlite:///:memory:")
        store = SqliteStateStore(run_id="test", engine=engine)
        ```
    """

    def __init__(
        self,
        run_id: str,
        state_type: Type[MODEL_T] | None = None,
        engine: Engine | None = None,
        serializer: BaseSerializer | None = None,
    ) -> None:
        """Initialize SQLite state store.

        Args:
            run_id: Unique identifier for this workflow run.
            state_type: Pydantic model type for state. Defaults to DictState.
            engine: SQLAlchemy engine for SQLite.
            serializer: Serializer for state values. Defaults to JsonSerializer.
        """
        # SQLite doesn't support schemas
        super().__init__(
            run_id=run_id,
            state_type=state_type,
            engine=engine,
            serializer=serializer,
            schema=None,
        )

    def _run_migrations(self) -> None:
        """Create the workflow_state table if it doesn't exist."""
        with self.engine.begin() as conn:
            # Check if table exists
            result = conn.execute(
                text(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name=:table_name
                    """
                ),
                {"table_name": self.TABLE_NAME},
            )
            exists = result.fetchone() is not None

            if not exists:
                # Create table
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE {self.TABLE_NAME} (
                            run_id TEXT PRIMARY KEY,
                            state_json TEXT NOT NULL,
                            state_type TEXT NOT NULL,
                            state_module TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        )
                        """  # noqa: S608
                    )
                )
                logger.info(f"Created table {self.TABLE_NAME} for workflow state")

    def _lock_row_for_update(self, conn: Connection) -> dict[str, Any] | None:
        """Lock using SQLite's exclusive transaction (BEGIN EXCLUSIVE).

        SQLite doesn't support SELECT FOR UPDATE, but the exclusive transaction
        mode provides database-level locking.
        """
        # Enable exclusive mode for this transaction
        # Note: This is handled at the connection level; here we just query
        result = conn.execute(
            text(
                f"""
                SELECT state_json, state_type, state_module
                FROM {self.TABLE_NAME}
                WHERE run_id = :run_id
                """  # noqa: S608
            ),
            {"run_id": self._run_id},
        )
        row = result.fetchone()

        if row is None:
            return None

        return {
            "state_json": row[0],
            "state_type": row[1],
            "state_module": row[2],
        }

    def _upsert_state(
        self,
        conn: Connection,
        state_json: str,
        state_type_name: str,
        state_module: str,
        now: datetime,
    ) -> None:
        """Perform SQLite upsert using INSERT ... ON CONFLICT."""
        # Use raw SQL to avoid DateTime type conversion issues with TEXT columns
        conn.execute(
            text(
                f"""
                INSERT INTO {self.TABLE_NAME} (run_id, state_json, state_type, state_module, created_at, updated_at)
                VALUES (:run_id, :state_json, :state_type, :state_module, :created_at, :updated_at)
                ON CONFLICT (run_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    state_type = excluded.state_type,
                    state_module = excluded.state_module,
                    updated_at = excluded.updated_at
                """  # noqa: S608
            ),
            {
                "run_id": self._run_id,
                "state_json": state_json,
                "state_type": state_type_name,
                "state_module": state_module,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            },
        )

    @classmethod
    def from_dict(
        cls,
        serialized_state: dict[str, Any],
        serializer: BaseSerializer,
    ) -> SqliteStateStore[Any]:
        """Restore a SQLite state store from serialized metadata.

        Note: The engine must be set separately after restoration.

        Args:
            serialized_state: Payload from to_dict.
            serializer: Serializer for state values.

        Returns:
            A new SqliteStateStore instance (engine must be set before use).
        """
        if not serialized_state:
            raise ValueError("Cannot restore SqliteStateStore from empty dict")

        run_id = serialized_state["run_id"]
        state_type_name = serialized_state.get("state_type", "DictState")
        state_module = serialized_state.get(
            "state_module", "workflows.context.state_store"
        )

        # Import the state type
        if state_type_name == "DictState":
            state_type: type[BaseModel] = DictState
        else:
            qualified_name = f"{state_module}.{state_type_name}"
            state_type = _import_class(qualified_name)

        return cls(
            run_id=run_id,
            state_type=state_type,  # type: ignore[arg-type]
            serializer=serializer,
        )


def create_state_store(
    run_id: str,
    engine: Engine,
    state_type: type[BaseModel] | None = None,
    serializer: BaseSerializer | None = None,
) -> SqlStateStore[Any]:
    """Factory function to create the appropriate state store for an engine.

    Automatically selects PostgresStateStore or SqliteStateStore based on
    the engine's dialect.

    Args:
        run_id: Unique identifier for this workflow run.
        engine: SQLAlchemy engine.
        state_type: Pydantic model type for state. Defaults to DictState.
        serializer: Serializer for state values. Defaults to JsonSerializer.

    Returns:
        Appropriate SqlStateStore subclass instance.

    Raises:
        ValueError: If the engine dialect is not supported.
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
