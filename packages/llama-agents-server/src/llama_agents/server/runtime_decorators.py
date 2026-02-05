# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""
Base decorator classes for Runtime, InternalRunAdapter, and ExternalRunAdapter.

These provide a simple forwarding pattern: accept an inner instance, delegate
every method to it. Subclasses override only the methods they need to customise.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator

from workflows.context.state_store import StateStore
from workflows.events import Event, StopEvent
from workflows.runtime.types.named_task import NamedTask
from workflows.runtime.types.plugin import (
    ExternalRunAdapter,
    InternalRunAdapter,
    RegisteredWorkflow,
    Runtime,
    WaitResult,
)
from workflows.runtime.types.ticks import WorkflowTick

if TYPE_CHECKING:
    from workflows.context.serializers import BaseSerializer
    from workflows.events import StartEvent
    from workflows.runtime.types.internal_state import BrokerState
    from workflows.workflow import Workflow


class BaseRuntimeDecorator(Runtime):
    """Decorator base for :class:`Runtime`.

    Wraps an inner runtime and forwards every call to it.  Subclasses can
    override individual methods to add behaviour (logging, metrics, auth,
    etc.) without re-implementing the full interface.
    """

    def __init__(self, inner: Runtime) -> None:
        self._inner = inner

    def register(self, workflow: Workflow) -> RegisteredWorkflow:
        return self._inner.register(workflow)

    def run_workflow(
        self,
        run_id: str,
        workflow: Workflow,
        init_state: BrokerState,
        start_event: StartEvent | None = None,
        serialized_state: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> ExternalRunAdapter:
        return self._inner.run_workflow(
            run_id,
            workflow,
            init_state,
            start_event=start_event,
            serialized_state=serialized_state,
            serializer=serializer,
        )

    def get_internal_adapter(self, workflow: Workflow) -> InternalRunAdapter:
        return self._inner.get_internal_adapter(workflow)

    def get_external_adapter(self, run_id: str) -> ExternalRunAdapter:
        return self._inner.get_external_adapter(run_id)

    def launch(self) -> None:
        self._inner.launch()

    def destroy(self) -> None:
        self._inner.destroy()

    def track_workflow(self, workflow: Workflow) -> None:
        self._inner.track_workflow(workflow)

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        return self._inner.get_registered(workflow)

    @contextmanager
    def registering(self) -> Generator[Runtime, None, None]:
        with self._inner.registering() as rt:
            yield rt


class BaseInternalRunAdapterDecorator(InternalRunAdapter):
    """Decorator base for :class:`InternalRunAdapter`.

    Wraps an inner adapter and forwards every call to it.  Subclasses can
    override individual methods to intercept or augment behaviour.
    """

    def __init__(self, inner: InternalRunAdapter) -> None:
        self._inner = inner

    @property
    def run_id(self) -> str:
        return self._inner.run_id

    async def write_to_event_stream(self, event: Event) -> None:
        await self._inner.write_to_event_stream(event)

    async def get_now(self) -> float:
        return await self._inner.get_now()

    async def send_event(self, tick: WorkflowTick) -> None:
        await self._inner.send_event(tick)

    async def wait_receive(
        self,
        timeout_seconds: float | None = None,
    ) -> WaitResult:
        return await self._inner.wait_receive(timeout_seconds)

    async def close(self) -> None:
        await self._inner.close()

    def get_state_store(self) -> StateStore[Any] | None:
        return self._inner.get_state_store()

    async def finalize_step(self) -> None:
        await self._inner.finalize_step()

    async def wait_for_next_task(
        self,
        task_set: list[NamedTask],
        timeout: float | None = None,
    ) -> asyncio.Task[Any] | None:
        return await self._inner.wait_for_next_task(task_set, timeout)


class BaseExternalRunAdapterDecorator(ExternalRunAdapter):
    """Decorator base for :class:`ExternalRunAdapter`.

    Wraps an inner adapter and forwards every call to it.  Subclasses can
    override individual methods to intercept or augment behaviour.
    """

    def __init__(self, inner: ExternalRunAdapter) -> None:
        self._inner = inner

    @property
    def run_id(self) -> str:
        return self._inner.run_id

    async def send_event(self, tick: WorkflowTick) -> None:
        await self._inner.send_event(tick)

    def stream_published_events(self) -> AsyncGenerator[Event, None]:
        return self._inner.stream_published_events()

    async def close(self) -> None:
        await self._inner.close()

    async def get_result(self) -> StopEvent:
        return await self._inner.get_result()

    async def cancel(self) -> None:
        await self._inner.cancel()

    def get_state_store(self) -> StateStore[Any] | None:
        return self._inner.get_state_store()
