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
from workflows.resource import (
    Resource,
    ResourceConfig,
    ResourceManager,
    _Resource,
    _ResourceConfig,
)
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


class FileData(BaseModel):
    file: str
    permission_mode: str


class FileOperator:
    def __init__(self, data: FileData) -> None:
        self.file = data.file
        self.permission_mode = data.permission_mode

    def operate(self) -> str:
        if self.permission_mode == "r":
            with open(self.file, self.permission_mode) as f:
                return f.read()
        elif self.permission_mode == "w":
            with open(self.file, self.permission_mode) as f:
                f.write("hello world!")
                return "hello world!"
        else:
            raise ValueError(f"Unsupported operation: {self.permission_mode}")


class ChatMessages(BaseModel):
    messages: list[str]


class Fs(BaseModel):
    files: list[str]
    dirs: list[str]


@pytest.mark.asyncio
async def test_function_resource_init() -> None:
    def get_string() -> str:
        return "string"

    retval = Resource(get_string)
    assert isinstance(retval, _Resource)
    assert "get_string" in retval.name
    assert retval.cache
    assert not retval._is_async

    result = await retval.call()
    assert result == "string"


def test_resource_config_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"messages": ["hello"]}

    with open("config.json", "w") as f:
        json.dump(data, f)

    retval = ResourceConfig(config_file="config.json")
    assert isinstance(retval, _ResourceConfig)
    assert retval.path_selector is None
    assert retval.config_file == "config.json"
    assert retval.cls_factory is None
    assert retval.cache
    assert retval.name == "config.json"

    # modify path selector, modify name
    retval.path_selector = "hello.world"
    assert retval.name == "config.json.hello.world"

    retval.path_selector = None

    with pytest.raises(
        ValueError,
        match="Class factory should be set to a BaseModel subclass before calling",
    ):
        retval.call()

    # define a cls_factory for the resource to be called
    retval.cls_factory = ChatMessages

    result = retval.call()
    assert isinstance(result, ChatMessages)
    assert result.messages == ["hello"]


def test_resource_config_path_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data = {
        "memory": {"messages": ["hello"]},
        "core": {"fs": {"files": ["hello.py"], "dirs": ["hello/"]}},
    }

    with open("config.json", "w") as f:
        json.dump(data, f)

    resource = ResourceConfig(config_file="config.json", path_selector="memory")
    assert resource.name == "config.json.memory"
    resource.cls_factory = ChatMessages
    value = resource.call()
    assert isinstance(value, ChatMessages)
    assert value.messages == ["hello"]
    resource.path_selector = "core.fs"
    assert resource.name == "config.json.core.fs"
    resource.cls_factory = Fs
    value = resource.call()
    assert isinstance(value, Fs)
    assert value.files == ["hello.py"]
    assert value.dirs == ["hello/"]


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
async def test_resource_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"file": "hello.py", "permission_mode": "r"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    with open("hello.py", "w") as f:
        f.write("print('hello')")

    def get_file_operator(
        config: Annotated[FileData, ResourceConfig(config_file="config.json")],
    ) -> FileOperator:
        return FileOperator(data=config)

    class TestWorkflow(Workflow):
        @step
        def start_step(self, ev: StartEvent) -> SecondEvent:
            print("Start step is done", flush=True)
            return SecondEvent(msg="Hello")

        @step
        def f1(
            self,
            ev: SecondEvent,
            file_operator: Annotated[FileOperator, Resource(get_file_operator)],
        ) -> StopEvent:
            assert file_operator.file == "hello.py"
            assert file_operator.permission_mode == "r"
            assert file_operator.operate() == "print('hello')"
            return StopEvent(result=None)

    wf = TestWorkflow(disable_validation=True)
    await wf.run()


@pytest.mark.asyncio
async def test_resource_manager_resource_configs_caching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"file": "hello.py", "permission_mode": "r"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    def get_file_operator(
        config: Annotated[FileData, ResourceConfig(config_file="config.json")],
    ) -> FileOperator:
        return FileOperator(data=config)

    manager = ResourceManager()

    resource = Resource(get_file_operator)

    val = await manager.get(resource)
    assert isinstance(val, FileOperator)
    assert val.file == data["file"]
    assert val.permission_mode == data["permission_mode"]

    cached_resource_config = manager.get_resource_config(
        resource_name=resource.name,
        resource_config=ResourceConfig(config_file="config.json"),
    )
    assert cached_resource_config is not None
    assert isinstance(cached_resource_config, FileData)
    assert cached_resource_config.file == data["file"]
    assert cached_resource_config.permission_mode == data["permission_mode"]

    all_cached_configs = manager.get_all_resource_configs(resource.name)
    assert isinstance(all_cached_configs, dict)
    assert len(all_cached_configs) == 1
    assert "config.json" in all_cached_configs
    assert isinstance(all_cached_configs["config.json"], FileData)


@pytest.mark.asyncio
async def test_resource_manager_resource_configs_not_caching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"file": "hello.py", "permission_mode": "r"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    def get_file_operator(
        config: Annotated[
            FileData, ResourceConfig(config_file="config.json", cache=False)
        ],
    ) -> FileOperator:
        return FileOperator(data=config)

    manager = ResourceManager()

    resource = Resource(get_file_operator)
    await manager.get(resource)
    cached_resource_config = manager.get_resource_config(
        resource_name=resource.name,
        resource_config=ResourceConfig(config_file="config.json"),
    )
    assert cached_resource_config is None  # check that it was not cached


@pytest.mark.asyncio
async def test_resource_manager_resource_configs_mixed_caching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"file": "hello.py", "permission_mode": "r"}
    with open("config.json", "w") as f:
        json.dump(data, f)

    # cache only the resource config, do not cache the resource itself
    def get_file_operator(
        config: Annotated[FileData, ResourceConfig(config_file="config.json")],
    ) -> FileOperator:
        return FileOperator(data=config)

    manager = ResourceManager()
    resource = Resource(get_file_operator, cache=False)

    await manager.get(resource)
    cached_resource_config = manager.get_resource_config(
        resource_name=resource.name,
        resource_config=ResourceConfig(config_file="config.json"),
    )
    assert cached_resource_config is not None
    assert isinstance(cached_resource_config, FileData)
    assert cached_resource_config.file == data["file"]
    assert cached_resource_config.permission_mode == data["permission_mode"]

    all_cached_configs = manager.get_all_resource_configs(resource.name)
    assert isinstance(all_cached_configs, dict)
    assert len(all_cached_configs) == 1
    assert "config.json" in all_cached_configs
    assert isinstance(all_cached_configs["config.json"], FileData)


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


async def test_caching_behavior_resource_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"file": "hello.py", "permission_mode": "r"}
    data_1 = {"file": "bye.py", "permission_mode": "w"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    def get_file_operator(
        config: Annotated[FileData, ResourceConfig(config_file="config.json")],
    ) -> FileOperator:
        return FileOperator(data=config)

    class TestWorkflow(Workflow):
        @step
        def start_step(
            self,
            ev: StartEvent,
            file: Annotated[FileOperator, Resource(get_file_operator)],
        ) -> SecondEvent:
            assert file.file == "hello.py"
            assert file.permission_mode == "r"
            # change config.json: the underlying resource has been cached, so will be unaffected
            with open("config.json", "w") as f:
                json.dump(data_1, f)
            return SecondEvent(msg="Hello")

        @step
        def f1(
            self,
            ev: SecondEvent,
            file: Annotated[FileOperator, Resource(get_file_operator)],
        ) -> StopEvent:
            # this resource has been cached,
            # so even if config.json has changed,
            # the resource remains the same
            assert file.file == "hello.py"
            assert file.permission_mode == "r"
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


async def test_non_caching_behavior_resource_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data = {"file": "hello.py", "permission_mode": "r"}
    data_1 = {"file": "bye.py", "permission_mode": "w"}

    with open("config.json", "w") as f:
        json.dump(data, f)

    def get_file_operator(
        config: Annotated[
            FileData, ResourceConfig(config_file="config.json", cache=False)
        ],
    ) -> FileOperator:
        return FileOperator(data=config)

    class TestWorkflow(Workflow):
        @step
        def start_step(
            self,
            ev: StartEvent,
            file: Annotated[FileOperator, Resource(get_file_operator, cache=False)],
        ) -> SecondEvent:
            assert file.file == "hello.py"
            assert file.permission_mode == "r"
            # change config.json: the underlying resource has not been cached, so it will be affected by the change
            with open("config.json", "w") as f:
                json.dump(data_1, f)
            return SecondEvent(msg="Hello")

        @step
        def f1(
            self,
            ev: SecondEvent,
            file: Annotated[FileOperator, Resource(get_file_operator, cache=False)],
        ) -> StopEvent:
            # this resource has not been cached,
            # so, since config.json has changed,
            # the resource changed too
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
