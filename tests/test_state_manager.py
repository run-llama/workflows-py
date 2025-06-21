import pytest
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing import Any

from workflows.context.serializers import JsonSerializer
from workflows.context.state_manager import DictState, InMemoryStateManager


class MyRandomObject:
    def __init__(self, attrib: str):
        self.attrib = attrib


class MyState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    my_obj: MyRandomObject
    name: str
    age: int

    @field_serializer("my_obj")
    def serialize_my_obj(self, my_obj: MyRandomObject, _info: Any) -> str:
        return my_obj.attrib

    @field_validator("my_obj", mode="before")
    @classmethod
    def deserialize_my_obj(cls, v: str) -> MyRandomObject:
        return MyRandomObject(attrib=v)


@pytest.fixture
def default_state_manager() -> InMemoryStateManager[DictState]:
    return InMemoryStateManager(DictState())


@pytest.fixture
def custom_state_manager() -> InMemoryStateManager[MyState]:
    return InMemoryStateManager(
        MyState(my_obj=MyRandomObject("llama-index"), name="John", age=30)
    )


@pytest.mark.asyncio
async def test_state_manager_defaults(
    default_state_manager: InMemoryStateManager[DictState],
) -> None:
    assert default_state_manager.state == DictState()

    await default_state_manager.set("name", "John")
    await default_state_manager.set("age", 30)

    assert await default_state_manager.get("name") == "John"
    assert await default_state_manager.get("age") == 30

    await default_state_manager.set("nested", {"a": "b"})
    assert await default_state_manager.get("nested.a") == "b"

    await default_state_manager.set("nested.a", "c")
    assert await default_state_manager.get("nested.a") == "c"

    full_state = await default_state_manager.get_all()
    assert full_state.name == "John"
    assert full_state.age == 30
    assert full_state.nested["a"] == "c"


@pytest.mark.asyncio
async def test_default_state_manager_serialization(
    default_state_manager: InMemoryStateManager[DictState],
) -> None:
    assert default_state_manager.state == DictState()

    await default_state_manager.set("name", "John")
    await default_state_manager.set("age", 30)

    assert await default_state_manager.get("name") == "John"
    assert await default_state_manager.get("age") == 30

    data = default_state_manager.to_dict(JsonSerializer())
    new_state_manager: InMemoryStateManager[DictState] = InMemoryStateManager.from_dict(
        data, JsonSerializer()
    )

    assert await new_state_manager.get("name") == "John"
    assert await new_state_manager.get("age") == 30
