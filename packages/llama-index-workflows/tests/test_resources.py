# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

import json
from pathlib import Path
from typing import Annotated, Optional
from unittest import mock

import pytest
from pydantic import BaseModel, Field
from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceManager, _ConfiguredResource, _Resource
from workflows.workflow import Workflow

# Global counters used in resource workflow tests
cc: int
cc1: int
cc2: int


class SecondEvent(Event):
    msg: str = Field(description="A message")


class ThirdEvent(Event):
    msg: str = Field(description="A message")


class ChatMessage(BaseModel):
    @classmethod
    def from_str(cls, role, content):  # type: ignore
        return mock.MagicMock(content=content)


class Memory(mock.MagicMock):
    @classmethod
    def from_defaults(cls, *args, **kwargs):  # type: ignore
        return mock.MagicMock()


class MessageStopEvent(StopEvent):
    llm_response: Optional[str] = Field(default=None)


@pytest.mark.asyncio
async def test_function_resource_init() -> None:
    def get_string() -> str:
        return "string"

    retval = Resource(get_string)
    assert isinstance(retval, _Resource)
    assert retval.type == "function"
    assert "get_string" in retval.name
    assert retval.cache
    assert not retval._is_async

    result = await retval.call()
    assert result == "string"


@pytest.mark.asyncio
async def test_pydantic_resource_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"messages": ["hello"]}

    class Memory(BaseModel):
        messages: list[str]

    with open("config.json", "w") as f:
        json.dump(data, f)

    retval = Resource(config_file="config.json")
    assert isinstance(retval, _ConfiguredResource)
    assert retval.type == "configured"
    assert retval.config_file == "config.json"
    assert retval.cls_factory is None
    assert retval.cache
    assert retval.name == "config.json"

    with pytest.raises(
        ValueError,
        match="Class factory should be set to a BaseModel subclass before calling",
    ):
        await retval.call()

    # define a cls_factory for the resource to be called
    retval.cls_factory = Memory

    result = await retval.call()
    assert isinstance(result, Memory)
    assert result.messages == ["hello"]


def test_resource_init_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError,
        match="At least one between `factory` and `config_file` has to be provided",
    ):
        Resource()

    with open("config.md", "w") as f:
        f.write("# Hello")

    with pytest.raises(
        ValueError,
        match="Only JSON files can be used to load Pydantic-based resources.",
    ):
        Resource(config_file="config.md")

    with pytest.raises(FileNotFoundError, match="No such file: config.json"):
        Resource(config_file="config.json")


@pytest.mark.asyncio
async def test_resource() -> None:
    m = Memory.from_defaults("user_id_123", token_limit=60000)

    def get_memory(*args, **kwargs) -> Memory:  # type: ignore
        return m

    class TestWorkflow(Workflow):
        @step
        def start_step(self, ev: StartEvent) -> SecondEvent:
            print("Start step is done", flush=True)
            return SecondEvent(msg="Hello")

        @step
        def f1(
            self, ev: SecondEvent, memory: Annotated[Memory, Resource(get_memory)]
        ) -> StopEvent:
            memory.put(ChatMessage.from_str(role="user", content=ev.msg))
            return StopEvent()

    wf = TestWorkflow(disable_validation=True)
    await wf.run()
    m.put.assert_called_once()


@pytest.mark.asyncio
async def test_resource_async() -> None:
    m = Memory.from_defaults("user_id_123", token_limit=60000)

    async def get_memory(*args, **kwargs) -> Memory:  # type: ignore
        return m

    class TestWorkflow(Workflow):
        @step
        def start_step(self, ev: StartEvent) -> SecondEvent:
            print("Start step is done", flush=True)
            return SecondEvent(msg="Hello")

        @step
        def f1(
            self,
            ev: SecondEvent,
            history: Annotated[Memory, Resource(get_memory)],
        ) -> StopEvent:
            history.put(ChatMessage.from_str(role="user", content=ev.msg))
            return StopEvent()

    wf = TestWorkflow(disable_validation=True)
    await wf.run()
    m.put.assert_called_once()


@pytest.mark.asyncio
async def test_resource_pydantic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    class FileData(BaseModel):
        file: str
        permission_mode: str

    data = {"file": "hello.py", "permission_mode": "r"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    with open("hello.py", "w") as f:
        f.write("print('hello')")

    class TestWorkflow(Workflow):
        @step
        def start_step(self, ev: StartEvent) -> SecondEvent:
            print("Start step is done", flush=True)
            return SecondEvent(msg="Hello")

        @step
        def f1(
            self,
            ev: SecondEvent,
            file: Annotated[FileData, Resource(config_file="config.json")],
        ) -> StopEvent:
            with open(file.file, file.permission_mode) as f:
                assert f.readable()
                content = f.read()
            assert content == "print('hello')"
            return StopEvent()

    wf = TestWorkflow(disable_validation=True)
    await wf.run()


@pytest.mark.asyncio
async def test_resource_manager_pydantic_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    class FileData(BaseModel):
        file: str
        permission_mode: str

    data = {"file": "hello.py", "permission_mode": "r"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    manager = ResourceManager()

    resource = Resource(config_file="config.json")
    assert isinstance(resource, _ConfiguredResource)

    with pytest.raises(
        ValueError,
        match="Class factory should be set to a BaseModel subclass before calling",
    ):
        await manager.get(resource)

    resource.cls_factory = FileData
    result = await manager.get(resource)
    assert isinstance(result, FileData)
    assert result.file == "hello.py"
    assert result.permission_mode == "r"

    # this resource has been cached now, so we can create a second one
    # with the same config_file and retrieve a FileData result
    # even without passing a cls_factory, under the assumption:
    # same config_file -> same resource that needs to be accessed
    resource_1 = Resource(config_file="config.json")
    result_1 = await manager.get(resource_1)
    assert isinstance(result_1, FileData)
    assert result_1.file == "hello.py"
    assert result_1.permission_mode == "r"


@pytest.mark.asyncio
async def test_caching_behavior() -> None:
    class CounterThing:
        counter = 0

        def incr(self) -> None:
            self.counter += 1

    class StepEvent(Event):
        pass

    def provide_counter_thing() -> CounterThing:
        return CounterThing()

    class TestWorkflow(Workflow):
        @step
        async def test_step(
            self,
            ev: StartEvent,
            counter_thing: Annotated[CounterThing, Resource(provide_counter_thing)],
        ) -> StepEvent:
            counter_thing.incr()
            return StepEvent()

        @step
        async def test_step_2(
            self,
            ev: StepEvent,
            counter_thing: Annotated[CounterThing, Resource(provide_counter_thing)],
        ) -> StopEvent:
            global cc
            counter_thing.incr()
            cc = counter_thing.counter  # type: ignore
            return StopEvent()

    wf_1 = TestWorkflow(disable_validation=True)
    await wf_1.run()
    assert (
        cc == 2  # type: ignore
    )  # this is expected to be 2, as it is a cached resource shared by test_step and test_step_2, which means at test_step counter_thing.counter goes from 0 to 1 and at test_step_2 goes from 1 to 2

    wf_2 = TestWorkflow(disable_validation=True)
    await wf_2.run()
    assert (
        cc == 2  # type: ignore
    )  # the cache is workflow-specific, so since wf_2 is different from wf_1, we expect no interference between the two


async def test_caching_behavior_pydantic_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    class FileData(BaseModel):
        file: str
        permission_mode: str

    data = {"file": "hello.py", "permission_mode": "r"}
    data_1 = {"file": "bye.py", "permission_mode": "w"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    with open("config_1.json", "w") as f:
        json.dump(data_1, f)

    class TestWorkflow(Workflow):
        @step
        def start_step(
            self,
            ev: StartEvent,
            file: Annotated[FileData, Resource(config_file="config.json")],
        ) -> SecondEvent:
            assert file.file == "hello.py"
            assert file.permission_mode == "r"
            # change config.json, file has been cached, so will be unaffected
            with open("config.json", "w") as f:
                json.dump(data_1, f)
            return SecondEvent(msg="Hello")

        @step
        def f1(
            self,
            ev: SecondEvent,
            file: Annotated[FileData, Resource(config_file="config_1.json")],
            other_file: Annotated[FileData, Resource(config_file="config.json")],
        ) -> StopEvent:
            # even if the resource requested is of the same _type_
            # the config is different
            # so caching here has no effect
            assert file.file == "bye.py"
            assert file.permission_mode == "w"
            # this resource has been cached,
            # so even if config.json has changed,
            # the resource remains the same
            assert other_file.file == "hello.py"
            assert other_file.permission_mode == "r"
            return StopEvent()

    wf = TestWorkflow(disable_validation=True)
    await wf.run()


@pytest.mark.asyncio
async def test_non_caching_behavior() -> None:
    class CounterThing:
        counter = 0

        def incr(self) -> None:
            self.counter += 1

    class StepEvent(Event):
        pass

    def provide_counter_thing() -> CounterThing:
        return CounterThing()

    class TestWorkflow(Workflow):
        @step
        async def test_step(
            self,
            ev: StartEvent,
            counter_thing: Annotated[CounterThing, Resource(provide_counter_thing)],
        ) -> StepEvent:
            global cc1
            counter_thing.incr()
            cc1 = counter_thing.counter  # type: ignore
            return StepEvent()

        @step
        async def test_step_2(
            self,
            ev: StepEvent,
            counter_thing: Annotated[
                CounterThing, Resource(provide_counter_thing, cache=False)
            ],
        ) -> StopEvent:
            global cc2
            counter_thing.incr()
            cc2 = counter_thing.counter  # type: ignore
            return StopEvent()

    wf_1 = TestWorkflow(disable_validation=True)
    await wf_1.run()
    assert cc1 == 1  # type: ignore
    assert cc2 == 1  # type: ignore


async def test_non_caching_behavior_pydantic_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    class FileData(BaseModel):
        file: str
        permission_mode: str

    data = {"file": "hello.py", "permission_mode": "r"}
    data_1 = {"file": "bye.py", "permission_mode": "w"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    class TestWorkflow(Workflow):
        @step
        def start_step(
            self,
            ev: StartEvent,
            file: Annotated[FileData, Resource(config_file="config.json", cache=False)],
        ) -> SecondEvent:
            assert file.file == "hello.py"
            assert file.permission_mode == "r"
            # change config.json: file has not been cached, so it will be affected in the next step
            with open("config.json", "w") as f:
                json.dump(data_1, f)
            return SecondEvent(msg="Hello")

        @step
        def f1(
            self,
            ev: SecondEvent,
            file: Annotated[FileData, Resource(config_file="config.json", cache=False)],
        ) -> StopEvent:
            # This resource was not cached at first
            # so it has been affected by the change in
            # the first step
            assert file.file == "bye.py"
            assert file.permission_mode == "w"
            return StopEvent()

    wf = TestWorkflow(disable_validation=True)
    await wf.run()


@pytest.mark.asyncio
async def test_resource_manager() -> None:
    m = ResourceManager()
    await m.set("test_resource", 42)
    assert m.get_all() == {"test_resource": 42}
