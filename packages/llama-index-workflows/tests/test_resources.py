# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from typing import Annotated, Optional
from unittest import mock

import pytest
from pydantic import BaseModel, Field

from workflows.decorators import step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceManager
from workflows.workflow import Workflow


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
    )  # this is expected to be 2, as it is a cached resource shared by test_step and test_step_2, which means at test_step it counter_thing.counter goes from 0 to 1 and at test_step_2 goes from 1 to 2

    wf_2 = TestWorkflow(disable_validation=True)
    await wf_2.run()
    assert (
        cc == 2  # type: ignore
    )  # the cache is workflow-specific, so since wf_2 is different from wf_1, we expect no interference between the two


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


@pytest.mark.asyncio
async def test_resource_manager() -> None:
    m = ResourceManager()
    await m.set("test_resource", 42)
    assert m.get_all() == {"test_resource": 42}
