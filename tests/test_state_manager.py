import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_serializer,
    field_validator,
)
from typing import Union

from workflows.context.serializers import JsonSerializer
from workflows.context.state_store import DictState, InMemoryStateStore


class MyRandomObject:
    def __init__(self, name: str):
        self.name = name


class MyState(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        strict=True,
    )

    my_obj: MyRandomObject
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


@pytest.fixture
def default_state_manager() -> InMemoryStateStore[DictState]:
    return InMemoryStateStore(DictState())


@pytest.fixture
def custom_state_manager() -> InMemoryStateStore[MyState]:
    return InMemoryStateStore(
        MyState(
            my_obj=MyRandomObject("llama-index"),
            name="John",
            age=30,
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
async def test_custom_state_manager(
    custom_state_manager: InMemoryStateStore[MyState],
) -> None:
    assert (await custom_state_manager.get_state()).model_dump(mode="json") == MyState(
        my_obj=MyRandomObject("llama-index"), name="John", age=30
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
    new_state_manager: InMemoryStateStore[MyState] = InMemoryStateStore.from_dict(
        data, JsonSerializer()
    )

    assert await new_state_manager.get("name") == "Jane"
    assert await new_state_manager.get("age") == 25

    assert (await new_state_manager.get("my_obj")).name == "llama-index"
