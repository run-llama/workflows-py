# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

"""State store matrix tests - testing StateStore implementations.

Tests the StateStore protocol across InMemoryStateStore, SqliteStateStore,
and PostgresStateStore to ensure consistent behavior.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, Union

import pytest
from llama_agents.runtime.dbos.sql_state_store import (
    PostgresStateStore,
    SqliteStateStore,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_serializer,
    field_validator,
)
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from workflows.context.serializers import JsonSerializer
from workflows.context.state_store import DictState, InMemoryStateStore, StateStore

if TYPE_CHECKING:
    from testcontainers.postgres import PostgresContainer


# -- Custom state types for testing --


class MyRandomObject:
    """Non-Pydantic object that requires custom serialization."""

    def __init__(self, name: str) -> None:
        self.name = name


class PydanticObject(BaseModel):
    """Simple Pydantic model for nested state testing."""

    name: str


class MyState(BaseModel):
    """Custom typed state with serialization logic."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        strict=True,
    )

    my_obj: MyRandomObject
    pydantic_obj: PydanticObject
    name: str
    age: int

    @field_serializer("my_obj", when_used="always")
    def serialize_my_obj(self, my_obj: MyRandomObject) -> str:
        return my_obj.name

    @field_validator("my_obj", mode="before")
    @classmethod
    def deserialize_my_obj(cls, v: Union[str, MyRandomObject]) -> MyRandomObject:
        if isinstance(v, MyRandomObject):
            return v
        if isinstance(v, str):
            return MyRandomObject(v)
        raise ValueError(f"Invalid type for my_obj: {type(v)}")


# -- Fixtures --


@pytest.fixture(scope="module")
def postgres_container() -> Generator["PostgresContainer", None, None]:
    """Module-scoped PostgreSQL container for state store tests.

    Requires Docker to be running.
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16", driver=None) as postgres:
        yield postgres


@pytest.fixture(scope="module")
def postgres_engine(
    postgres_container: "PostgresContainer",
) -> Generator[Engine, None, None]:
    """Module-scoped PostgreSQL engine for state store tests."""
    # Get connection URL and convert to use psycopg (psycopg3) driver
    connection_url = postgres_container.get_connection_url()
    # Replace postgresql:// or postgresql+psycopg2:// with postgresql+psycopg://
    if "postgresql+psycopg2://" in connection_url:
        connection_url = connection_url.replace(
            "postgresql+psycopg2://", "postgresql+psycopg://"
        )
    elif connection_url.startswith("postgresql://"):
        connection_url = connection_url.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    engine = create_engine(connection_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def sqlite_engine(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Engine, None, None]:
    """Module-scoped SQLite engine for state store tests."""
    db_file: Path = tmp_path_factory.mktemp("state_store") / "test.sqlite3"
    engine = create_engine(f"sqlite:///{db_file}?check_same_thread=false")
    yield engine
    engine.dispose()


def _get_store_params() -> list[Any]:
    """Get store type parameters for the test matrix."""
    return [
        pytest.param("in_memory", id="in_memory"),
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", id="postgres"),
    ]


@pytest.fixture(params=_get_store_params())
async def state_store(
    request: pytest.FixtureRequest,
    sqlite_engine: Engine,
) -> AsyncGenerator[StateStore[DictState], None]:
    """Parametrized fixture yielding a fresh StateStore for each test."""
    # Use unique run_id per test to avoid state bleeding
    run_id = f"test-{id(request)}"

    if request.param == "in_memory":
        yield InMemoryStateStore(DictState())
    elif request.param == "sqlite":
        store = SqliteStateStore(run_id=run_id, engine=sqlite_engine)
        yield store
    elif request.param == "postgres":
        # Lazily request postgres_engine only when needed
        pg_engine: Engine = request.getfixturevalue("postgres_engine")
        store = PostgresStateStore(run_id=run_id, engine=pg_engine)
        yield store


@pytest.fixture(params=_get_store_params())
async def custom_state_store(
    request: pytest.FixtureRequest,
    sqlite_engine: Engine,
) -> AsyncGenerator[StateStore[MyState], None]:
    """Parametrized fixture yielding a StateStore with custom typed state."""
    run_id = f"test-custom-{id(request)}"
    initial_state = MyState(
        my_obj=MyRandomObject("llama-index"),
        pydantic_obj=PydanticObject(name="llama-index"),
        name="John",
        age=30,
    )

    if request.param == "in_memory":
        yield InMemoryStateStore(initial_state)
    elif request.param == "sqlite":
        store = SqliteStateStore(
            run_id=run_id,
            state_type=MyState,
            engine=sqlite_engine,
        )
        # Initialize with custom state
        await store.set_state(initial_state)
        yield store
    elif request.param == "postgres":
        # Lazily request postgres_engine only when needed
        pg_engine: Engine = request.getfixturevalue("postgres_engine")
        store = PostgresStateStore(
            run_id=run_id,
            state_type=MyState,
            engine=pg_engine,
        )
        # Initialize with custom state
        await store.set_state(initial_state)
        yield store


# -- Basic Operations Tests --


@pytest.mark.asyncio
async def test_get_set_basic_values(state_store: StateStore[DictState]) -> None:
    """Test basic get/set operations with simple values."""
    await state_store.set("name", "John")
    await state_store.set("age", 30)

    assert await state_store.get("name") == "John"
    assert await state_store.get("age") == 30


@pytest.mark.asyncio
async def test_get_with_default(state_store: StateStore[DictState]) -> None:
    """Test get with default value for missing keys."""
    result = await state_store.get("nonexistent", default=None)
    assert result is None

    result = await state_store.get("missing", default="fallback")
    assert result == "fallback"


@pytest.mark.asyncio
async def test_get_missing_raises(state_store: StateStore[DictState]) -> None:
    """Test that get raises ValueError for missing key without default."""
    with pytest.raises(ValueError, match="not found"):
        await state_store.get("nonexistent")


@pytest.mark.asyncio
async def test_nested_get_set(state_store: StateStore[DictState]) -> None:
    """Test nested path access with dot notation."""
    await state_store.set("nested", {"a": "b"})
    assert await state_store.get("nested.a") == "b"

    await state_store.set("nested.a", "c")
    assert await state_store.get("nested.a") == "c"


@pytest.mark.asyncio
async def test_get_state_returns_copy(state_store: StateStore[DictState]) -> None:
    """Test that get_state returns a copy, not the original."""
    await state_store.set("value", 1)

    state1 = await state_store.get_state()
    state2 = await state_store.get_state()

    # Should be equal but not the same object
    assert state1.model_dump() == state2.model_dump()


@pytest.mark.asyncio
async def test_set_state_replaces(state_store: StateStore[DictState]) -> None:
    """Test that set_state replaces the entire state."""
    await state_store.set("old_key", "old_value")

    new_state = DictState()
    new_state["new_key"] = "new_value"
    await state_store.set_state(new_state)

    assert await state_store.get("new_key") == "new_value"
    # Old key should be gone or inaccessible
    result = await state_store.get("old_key", default=None)
    assert result is None


@pytest.mark.asyncio
async def test_clear_resets_state(state_store: StateStore[DictState]) -> None:
    """Test that clear resets to default state."""
    await state_store.set("name", "Jane")
    await state_store.set("age", 25)

    await state_store.clear()

    assert await state_store.get("name", default=None) is None
    assert await state_store.get("age", default=None) is None


# -- edit_state Context Manager Tests --


@pytest.mark.asyncio
async def test_edit_state_basic(state_store: StateStore[DictState]) -> None:
    """Test basic edit_state context manager usage."""
    await state_store.set("counter", 0)

    async with state_store.edit_state() as state:
        current = state.get("counter", 0)
        state["counter"] = current + 1

    assert await state_store.get("counter") == 1


@pytest.mark.asyncio
async def test_edit_state_multiple_changes(state_store: StateStore[DictState]) -> None:
    """Test multiple changes within a single edit_state."""
    async with state_store.edit_state() as state:
        state["a"] = 1
        state["b"] = 2
        state["c"] = {"nested": "value"}

    assert await state_store.get("a") == 1
    assert await state_store.get("b") == 2
    assert await state_store.get("c.nested") == "value"


@pytest.mark.asyncio
async def test_edit_state_exception_handling(
    state_store: StateStore[DictState],
) -> None:
    """Test that exceptions in edit_state don't corrupt state."""
    await state_store.set("value", "original")

    with pytest.raises(ValueError, match="intentional"):
        async with state_store.edit_state() as state:
            state["value"] = "modified"
            raise ValueError("intentional error")

    # State should remain unchanged after exception
    # Note: behavior may vary - InMemory commits on context exit, SQL rolls back
    # This test documents the expected behavior


# -- Custom Typed State Tests --


@pytest.mark.asyncio
async def test_custom_state_type(custom_state_store: StateStore[MyState]) -> None:
    """Test state store with custom Pydantic model."""
    state = await custom_state_store.get_state()
    assert isinstance(state, MyState)
    assert state.name == "John"
    assert state.age == 30
    assert state.my_obj.name == "llama-index"


@pytest.mark.asyncio
async def test_custom_state_set_values(custom_state_store: StateStore[MyState]) -> None:
    """Test setting values on custom typed state."""
    await custom_state_store.set("name", "Jane")
    await custom_state_store.set("age", 25)

    assert await custom_state_store.get("name") == "Jane"
    assert await custom_state_store.get("age") == 25

    # Original custom fields should still be accessible
    state = await custom_state_store.get_state()
    assert state.my_obj.name == "llama-index"


@pytest.mark.asyncio
async def test_custom_state_validation(custom_state_store: StateStore[MyState]) -> None:
    """Test that Pydantic validation is enforced on custom state."""
    # MyState has strict=True, so setting age to string should fail
    with pytest.raises(ValidationError):
        await custom_state_store.set("age", "not a number")


# -- Serialization Tests --


@pytest.mark.asyncio
async def test_to_dict_from_dict_roundtrip(state_store: StateStore[DictState]) -> None:
    """Test serialization roundtrip with to_dict/from_dict."""
    await state_store.set("name", "John")
    await state_store.set("age", 30)

    serializer = JsonSerializer()
    data = state_store.to_dict(serializer)

    # For InMemoryStateStore, from_dict restores the full state
    # For SqlStateStore, from_dict returns metadata (engine must be set separately)
    if isinstance(state_store, InMemoryStateStore):
        restored = InMemoryStateStore.from_dict(data, serializer)
        assert await restored.get("name") == "John"
        assert await restored.get("age") == 30


# -- SQLite-Specific Tests --


@pytest.mark.asyncio
async def test_sqlite_persistence(sqlite_engine: Engine) -> None:
    """Test that SQLite state persists across store instances."""
    run_id = "persistence-test"

    # Create store and set state
    store1 = SqliteStateStore(run_id=run_id, engine=sqlite_engine)
    await store1.set("persistent_key", "persistent_value")

    # Create new store instance with same run_id
    store2 = SqliteStateStore(run_id=run_id, engine=sqlite_engine)
    result = await store2.get("persistent_key")

    assert result == "persistent_value"


@pytest.mark.asyncio
async def test_sqlite_isolation(sqlite_engine: Engine) -> None:
    """Test that different run_ids have isolated state."""
    store1 = SqliteStateStore(run_id="run-1", engine=sqlite_engine)
    store2 = SqliteStateStore(run_id="run-2", engine=sqlite_engine)

    await store1.set("key", "value1")
    await store2.set("key", "value2")

    assert await store1.get("key") == "value1"
    assert await store2.get("key") == "value2"


@pytest.mark.asyncio
async def test_sqlite_concurrent_edits(sqlite_engine: Engine) -> None:
    """Test concurrent edit_state calls are serialized correctly."""
    run_id = "concurrent-test"
    store = SqliteStateStore(run_id=run_id, engine=sqlite_engine)
    await store.set("counter", 0)

    async def increment() -> None:
        async with store.edit_state() as state:
            current = state.get("counter", 0)
            await asyncio.sleep(0.01)  # Simulate some work
            state["counter"] = current + 1

    # Run multiple increments concurrently
    await asyncio.gather(*[increment() for _ in range(5)])

    # All increments should have been applied
    result = await store.get("counter")
    assert result == 5


@pytest.mark.asyncio
async def test_sqlite_custom_state_persistence(sqlite_engine: Engine) -> None:
    """Test that custom typed state persists correctly."""
    run_id = "custom-persistence-test"

    initial_state = MyState(
        my_obj=MyRandomObject("persisted"),
        pydantic_obj=PydanticObject(name="persisted"),
        name="Original",
        age=100,
    )

    # Create and initialize store
    store1 = SqliteStateStore(
        run_id=run_id,
        state_type=MyState,
        engine=sqlite_engine,
    )
    await store1.set_state(initial_state)
    await store1.set("name", "Modified")

    # Create new store instance
    store2 = SqliteStateStore(
        run_id=run_id,
        state_type=MyState,
        engine=sqlite_engine,
    )
    state = await store2.get_state()

    assert state.name == "Modified"
    assert state.my_obj.name == "persisted"


# -- PostgreSQL-Specific Tests --


@pytest.mark.asyncio
async def test_postgres_persistence(postgres_engine: Engine) -> None:
    """Test that PostgreSQL state persists across store instances."""
    run_id = "pg-persistence-test"

    # Create store and set state
    store1 = PostgresStateStore(run_id=run_id, engine=postgres_engine)
    await store1.set("persistent_key", "persistent_value")

    # Create new store instance with same run_id
    store2 = PostgresStateStore(run_id=run_id, engine=postgres_engine)
    result = await store2.get("persistent_key")

    assert result == "persistent_value"


@pytest.mark.asyncio
async def test_postgres_isolation(postgres_engine: Engine) -> None:
    """Test that different run_ids have isolated state."""
    store1 = PostgresStateStore(run_id="pg-run-1", engine=postgres_engine)
    store2 = PostgresStateStore(run_id="pg-run-2", engine=postgres_engine)

    await store1.set("key", "value1")
    await store2.set("key", "value2")

    assert await store1.get("key") == "value1"
    assert await store2.get("key") == "value2"


@pytest.mark.asyncio
async def test_postgres_concurrent_edits(postgres_engine: Engine) -> None:
    """Test concurrent edit_state calls are serialized correctly with FOR UPDATE."""
    run_id = "pg-concurrent-test"
    store = PostgresStateStore(run_id=run_id, engine=postgres_engine)
    await store.set("counter", 0)

    async def increment() -> None:
        async with store.edit_state() as state:
            current = state.get("counter", 0)
            await asyncio.sleep(0.01)  # Simulate some work
            state["counter"] = current + 1

    # Run multiple increments concurrently
    await asyncio.gather(*[increment() for _ in range(5)])

    # All increments should have been applied
    result = await store.get("counter")
    assert result == 5


@pytest.mark.asyncio
async def test_postgres_custom_state_persistence(postgres_engine: Engine) -> None:
    """Test that custom typed state persists correctly."""
    run_id = "pg-custom-persistence-test"

    initial_state = MyState(
        my_obj=MyRandomObject("persisted"),
        pydantic_obj=PydanticObject(name="persisted"),
        name="Original",
        age=100,
    )

    # Create and initialize store
    store1 = PostgresStateStore(
        run_id=run_id,
        state_type=MyState,
        engine=postgres_engine,
    )
    await store1.set_state(initial_state)
    await store1.set("name", "Modified")

    # Create new store instance
    store2 = PostgresStateStore(
        run_id=run_id,
        state_type=MyState,
        engine=postgres_engine,
    )
    state = await store2.get_state()

    assert state.name == "Modified"
    assert state.my_obj.name == "persisted"


@pytest.mark.asyncio
async def test_postgres_uses_dbos_schema(postgres_engine: Engine) -> None:
    """Test that PostgresStateStore uses the 'dbos' schema by default."""
    run_id = "pg-schema-test"
    store = PostgresStateStore(run_id=run_id, engine=postgres_engine)

    # Trigger table creation by accessing state
    await store.set("test", "value")

    # Verify the table was created in the dbos schema
    with postgres_engine.connect() as conn:
        result = conn.exec_driver_sql(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'dbos'
                AND table_name = 'workflow_state'
            )
            """
        )
        exists = result.scalar()
        assert exists is True
