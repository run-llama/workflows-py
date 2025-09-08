import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_serializer,
    field_validator,
)
from typing import Union, cast, Type, Any

from workflows.context.serializers import JsonSerializer, BaseSerializer
from workflows.context.state_store import DictState, InMemoryStateStore


class MyRandomObject:
    def __init__(self, name: str):
        self.name = name


class PydanticObject(BaseModel):
    name: str


class MyState(BaseModel):
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


class MyUnserializableState(BaseModel):
    serializer_type: Type[BaseSerializer]
    test_data: dict[str, Any]


@pytest.fixture
def default_state_manager() -> InMemoryStateStore[DictState]:
    return InMemoryStateStore(DictState())


@pytest.fixture
def custom_state_manager() -> InMemoryStateStore[MyState]:
    return InMemoryStateStore(
        MyState(
            my_obj=MyRandomObject("llama-index"),
            pydantic_obj=PydanticObject(name="llama-index"),
            name="John",
            age=30,
        )
    )


@pytest.fixture
def unser_custom_state_manager() -> InMemoryStateStore[MyUnserializableState]:
    return InMemoryStateStore(
        MyUnserializableState(
            serializer_type=type(JsonSerializer()), test_data={"test": 1, "data": 2}
        )
    )


@pytest.mark.asyncio
async def test_state_manager_defaults(
    default_state_manager: InMemoryStateStore[DictState],
) -> None:
    assert (
        await default_state_manager.get_state()
    ).model_dump_json() == DictState().model_dump_json()

    await default_state_manager.set("name", "John")
    await default_state_manager.set("age", 30)

    assert await default_state_manager.get("name") == "John"
    assert await default_state_manager.get("age") == 30

    await default_state_manager.set("nested", {"a": "b"})
    assert await default_state_manager.get("nested.a") == "b"

    await default_state_manager.set("nested.a", "c")
    assert await default_state_manager.get("nested.a") == "c"

    full_state = await default_state_manager.get_state()
    assert full_state.name == "John"
    assert full_state.age == 30
    assert full_state.nested["a"] == "c"


@pytest.mark.asyncio
async def test_default_state_manager_serialization(
    default_state_manager: InMemoryStateStore[DictState],
) -> None:
    assert (
        await default_state_manager.get_state()
    ).model_dump() == DictState().model_dump()

    await default_state_manager.set("name", "John")
    await default_state_manager.set("age", 30)

    assert await default_state_manager.get("name") == "John"
    assert await default_state_manager.get("age") == 30

    data = default_state_manager.to_dict(JsonSerializer())
    new_state_manager: InMemoryStateStore[DictState] = InMemoryStateStore.from_dict(
        data, JsonSerializer()
    )

    assert await new_state_manager.get("name") == "John"
    assert await new_state_manager.get("age") == 30


@pytest.mark.asyncio
async def test_custom_state_manager_snapshot(
    default_state_manager: InMemoryStateStore[DictState],
) -> None:
    await default_state_manager.set("serializer", type(JsonSerializer()))
    await default_state_manager.set("test_data", {"test": 1, "data": 2})

    assert await default_state_manager.get("serializer") is type(JsonSerializer())
    assert await default_state_manager.get("test_data") == {"test": 1, "data": 2}

    # prove that to_dict throws error when called on this object
    with pytest.raises(ValueError):
        default_state_manager.to_dict(JsonSerializer())

    dict_snapshot = default_state_manager.to_dict_snapshot(JsonSerializer())
    assert isinstance(dict_snapshot, dict)
    assert (
        "state_data" in dict_snapshot
        and "state_type" in dict_snapshot
        and "state_module" in dict_snapshot
    )
    assert dict_snapshot["state_data"] == {
        "serializer": "<class 'workflows.context.serializers.JsonSerializer'>",
        "test_data": '{"test": 1, "data": 2}',
    }
    assert dict_snapshot["state_type"] == "DictState"
    assert dict_snapshot["state_module"] == "workflows.context.state_store"


@pytest.mark.asyncio
async def test_default_state_manager_snapshot(
    unser_custom_state_manager: InMemoryStateStore[MyUnserializableState],
) -> None:
    assert await unser_custom_state_manager.get("serializer_type") is type(
        JsonSerializer()
    )
    assert await unser_custom_state_manager.get("test_data") == {"test": 1, "data": 2}

    # prove that to_dict throws error when called on this object
    with pytest.raises(ValueError):
        unser_custom_state_manager.to_dict(JsonSerializer())

    dict_snapshot = unser_custom_state_manager.to_dict_snapshot(JsonSerializer())
    assert isinstance(dict_snapshot, dict)
    assert (
        "state_data" in dict_snapshot
        and "state_type" in dict_snapshot
        and "state_module" in dict_snapshot
    )
    assert dict_snapshot["state_data"] == {
        "serializer_type": "<class 'workflows.context.serializers.JsonSerializer'>",
        "test_data": '{"test": 1, "data": 2}',
    }
    assert dict_snapshot["state_type"] == "MyUnserializableState"
    assert dict_snapshot["state_module"] == "tests.test_state_manager"


@pytest.mark.asyncio
async def test_custom_state_manager(
    custom_state_manager: InMemoryStateStore[MyState],
) -> None:
    assert (await custom_state_manager.get_state()).model_dump(mode="json") == MyState(
        my_obj=MyRandomObject("llama-index"),
        pydantic_obj=PydanticObject(name="llama-index"),
        name="John",
        age=30,
    ).model_dump(mode="json")

    await custom_state_manager.set("name", "Jane")
    await custom_state_manager.set("age", 25)

    assert await custom_state_manager.get("name") == "Jane"
    assert await custom_state_manager.get("age") == 25

    full_state = await custom_state_manager.get_state()
    assert isinstance(full_state, MyState)
    assert full_state.name == "Jane"
    assert full_state.age == 25
    assert full_state.my_obj.name == "llama-index"
    assert full_state.pydantic_obj.name == "llama-index"

    # Ensure pydantic is providing type safety
    with pytest.raises(ValidationError):
        await custom_state_manager.set("age", "30")

    with pytest.raises(AttributeError):
        await custom_state_manager.set("age.nested", "llama-index")


@pytest.mark.asyncio
async def test_state_manager_custom_serialization(
    custom_state_manager: InMemoryStateStore[MyState],
) -> None:
    await custom_state_manager.set("name", "Jane")
    await custom_state_manager.set("age", 25)

    assert await custom_state_manager.get("name") == "Jane"
    assert await custom_state_manager.get("age") == 25

    data = custom_state_manager.to_dict(JsonSerializer())
    new_state_manager: InMemoryStateStore[MyState] = cast(
        InMemoryStateStore[MyState],
        InMemoryStateStore.from_dict(data, JsonSerializer()),
    )

    assert await new_state_manager.get("name") == "Jane"
    assert await new_state_manager.get("age") == 25

    assert (await new_state_manager.get("my_obj")).name == "llama-index"

    state = await new_state_manager.get_state()
    assert state.pydantic_obj.name == "llama-index"


@pytest.mark.asyncio
async def test_state_manager_clear() -> None:
    state_manager = InMemoryStateStore(DictState())
    await state_manager.set("name", "Jane")
    await state_manager.set("age", 25)

    await state_manager.clear()
    assert await state_manager.get("name", default=None) is None
    assert await state_manager.get("age", default=None) is None
