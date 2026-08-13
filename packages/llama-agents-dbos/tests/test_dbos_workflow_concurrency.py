# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from dbos import DBOS, DBOSConfig
from llama_agents.dbos import DBOSRuntime
from workflows.decorators import step
from workflows.errors import WorkflowCancelledByUser
from workflows.events import StartEvent, StopEvent, WorkflowCancelledEvent
from workflows.workflow import Workflow


class _AdmissionGate:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._release = threading.Event()
        self.started: list[str] = []

    async def enter(self, run: str) -> None:
        with self._condition:
            self.started.append(run)
            self._condition.notify_all()
        await asyncio.to_thread(self._release.wait)

    def wait_for_started(self, count: int, timeout: float = 5.0) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: len(self.started) >= count,
                timeout=timeout,
            )

    def release(self) -> None:
        self._release.set()


def _dbos_config(db_path: Path, name: str) -> DBOSConfig:
    return {
        "name": name,
        "system_database_url": (
            f"sqlite+pysqlite:///{db_path}?check_same_thread=false"
        ),
        "run_admin_server": False,
    }  # type: ignore[return-value]


def _blocking_workflow_type(
    gate: _AdmissionGate,
    *,
    class_name: str,
) -> type[Workflow]:
    class BlockingWorkflow(Workflow):
        @step
        async def block(self, ev: StartEvent) -> StopEvent:
            run = ev.get("run")
            assert isinstance(run, str)
            await gate.enter(run)
            return StopEvent(result=run)

    BlockingWorkflow.__name__ = class_name
    BlockingWorkflow.__qualname__ = class_name
    return BlockingWorkflow


async def _run_limited_admission_case(tmp_path: Path) -> None:
    DBOS(
        config=_dbos_config(
            tmp_path / "limited-admission.sqlite3",
            "dbos-limited-admission-test",
        )
    )
    runtime = DBOSRuntime(polling_interval_sec=0.01)
    gate = _AdmissionGate()
    workflow = _blocking_workflow_type(
        gate,
        class_name="LimitedAdmissionWorkflow",
    )(
        runtime=runtime,
        workflow_name="tests.dbos.limited-admission",
        num_concurrent_runs=1,
        timeout=10,
    )
    handlers: dict[str, Any] = {}

    try:
        await runtime.launch()
        handlers = {
            run_id: workflow.run(run=run_id, run_id=run_id)
            for run_id in ("limited-first", "limited-second")
        }
        admitted = await asyncio.to_thread(gate.wait_for_started, 1)
        assert admitted
        assert len(gate.started) == 1

        queued_run = (
            "limited-second" if gate.started == ["limited-first"] else "limited-first"
        )
        status = await DBOS.get_workflow_status_async(queued_run)
        assert status is not None
        assert status.status == "ENQUEUED"

        gate.release()
        results = await asyncio.wait_for(
            asyncio.gather(*handlers.values()),
            timeout=10,
        )
        assert set(results) == {"limited-first", "limited-second"}
        assert set(gate.started) == {"limited-first", "limited-second"}
    finally:
        gate.release()
        if handlers:
            await asyncio.gather(*handlers.values(), return_exceptions=True)
        await runtime.destroy()


def test_limited_workflow_queues_excess_runs(tmp_path: Path) -> None:
    asyncio.run(_run_limited_admission_case(tmp_path))


async def _run_unlimited_admission_case(tmp_path: Path) -> None:
    DBOS(
        config=_dbos_config(
            tmp_path / "unlimited-admission.sqlite3",
            "dbos-unlimited-admission-test",
        )
    )
    runtime = DBOSRuntime(polling_interval_sec=0.01)
    gate = _AdmissionGate()
    workflow = _blocking_workflow_type(
        gate,
        class_name="UnlimitedAdmissionWorkflow",
    )(
        runtime=runtime,
        workflow_name="tests.dbos.unlimited-admission",
        timeout=10,
    )
    handlers: dict[str, Any] = {}

    try:
        await runtime.launch()
        handlers = {
            run_id: workflow.run(run=run_id, run_id=run_id)
            for run_id in ("unlimited-first", "unlimited-second")
        }
        admitted = await asyncio.to_thread(gate.wait_for_started, 2)
        assert admitted
        assert set(gate.started) == {"unlimited-first", "unlimited-second"}

        statuses = await asyncio.gather(
            *(DBOS.get_workflow_status_async(run_id) for run_id in handlers)
        )
        assert all(status is not None for status in statuses)
        assert all(status.status != "ENQUEUED" for status in statuses if status)

        gate.release()
        results = await asyncio.wait_for(
            asyncio.gather(*handlers.values()),
            timeout=10,
        )
        assert set(results) == {"unlimited-first", "unlimited-second"}
    finally:
        gate.release()
        if handlers:
            await asyncio.gather(*handlers.values(), return_exceptions=True)
        await runtime.destroy()


def test_unlimited_workflow_starts_runs_directly(tmp_path: Path) -> None:
    asyncio.run(_run_unlimited_admission_case(tmp_path))


async def _run_queue_declaration_case(tmp_path: Path) -> None:
    DBOS(
        config=_dbos_config(
            tmp_path / "queue-declaration.sqlite3",
            "dbos-queue-declaration-test",
        )
    )
    runtime = DBOSRuntime(polling_interval_sec=0.125)
    _blocking_workflow_type(
        _AdmissionGate(),
        class_name="LimitedQueueWorkflow",
    )(
        runtime=runtime,
        workflow_name="tests.dbos.limited-queue",
        num_concurrent_runs=3,
    )
    _blocking_workflow_type(
        _AdmissionGate(),
        class_name="UnlimitedQueueWorkflow",
    )(
        runtime=runtime,
        workflow_name="tests.dbos.unlimited-queue",
    )

    try:
        await runtime.launch()
        queues = {queue.name: queue for queue in runtime.workflow_queues}
        assert list(queues) == [
            "_llamaindex_workflow_queue:tests.dbos.limited-queue",
            "_llamaindex_workflow_queue:tests.dbos.unlimited-queue",
        ]
        assert (
            queues[
                "_llamaindex_workflow_queue:tests.dbos.limited-queue"
            ].worker_concurrency
            == 3
        )
        assert (
            queues[
                "_llamaindex_workflow_queue:tests.dbos.unlimited-queue"
            ].worker_concurrency
            is None
        )
    finally:
        await runtime.destroy()


def test_runtime_declares_queue_for_every_workflow(tmp_path: Path) -> None:
    asyncio.run(_run_queue_declaration_case(tmp_path))


async def _run_destroy_relaunch_case(tmp_path: Path) -> None:
    config = _dbos_config(
        tmp_path / "reused-queue.sqlite3",
        "dbos-reused-queue-test",
    )
    workflow_name = "tests.dbos.reused-queue"

    DBOS(config=config)
    first_runtime = DBOSRuntime(polling_interval_sec=0.01)
    _blocking_workflow_type(
        _AdmissionGate(),
        class_name="FirstReusedQueueWorkflow",
    )(
        runtime=first_runtime,
        workflow_name=workflow_name,
        num_concurrent_runs=1,
    )
    await first_runtime.launch()
    queue = first_runtime.workflow_queues[0]
    await first_runtime.destroy(destroy_dbos=True)

    DBOS(config=config)
    second_runtime = DBOSRuntime(polling_interval_sec=0.02)
    _blocking_workflow_type(
        _AdmissionGate(),
        class_name="SecondReusedQueueWorkflow",
    )(
        runtime=second_runtime,
        workflow_name=workflow_name,
        num_concurrent_runs=2,
    )

    try:
        await second_runtime.launch()
        assert second_runtime.workflow_queues == (queue,)
        assert queue.worker_concurrency == 2
        assert queue.polling_interval_sec == 0.02
    finally:
        await second_runtime.destroy(destroy_dbos=True)


def test_destroy_relaunch_reuses_queue_with_updated_limit(tmp_path: Path) -> None:
    asyncio.run(_run_destroy_relaunch_case(tmp_path))


async def _run_queued_cancellation_case(tmp_path: Path) -> None:
    DBOS(
        config=_dbos_config(
            tmp_path / "queued-cancellation.sqlite3",
            "dbos-queued-cancellation-test",
        )
    )
    runtime = DBOSRuntime(polling_interval_sec=0.01)
    gate = _AdmissionGate()

    class QueuedCancellationWorkflow(Workflow):
        @step
        async def block(self, ev: StartEvent) -> StopEvent:
            run = ev.get("run")
            assert isinstance(run, str)
            await gate.enter(run)
            await asyncio.sleep(0.2)
            return StopEvent(result=run)

    workflow = QueuedCancellationWorkflow(
        runtime=runtime,
        workflow_name="tests.dbos.queued-cancellation",
        num_concurrent_runs=1,
        timeout=10,
    )
    handlers: dict[str, Any] = {}

    try:
        await runtime.launch()
        handlers = {
            run_id: workflow.run(run=run_id, run_id=run_id)
            for run_id in ("cancel-first", "cancel-second")
        }
        admitted = await asyncio.to_thread(gate.wait_for_started, 1)
        assert admitted
        queued_run = (
            "cancel-second" if gate.started == ["cancel-first"] else "cancel-first"
        )
        queued_handler = handlers[queued_run]

        await queued_handler._external_adapter.cancel()

        status = await DBOS.get_workflow_status_async(queued_run)
        assert status is not None
        assert status.status == "ENQUEUED"
        gate.release()
        await handlers[gate.started[0]]
        with pytest.raises(WorkflowCancelledByUser):
            await queued_handler
        assert isinstance(queued_handler.get_stop_event(), WorkflowCancelledEvent)
    finally:
        gate.release()
        if handlers:
            await asyncio.gather(*handlers.values(), return_exceptions=True)
        await runtime.destroy()


def test_enqueued_cancellation_waits_for_admission(tmp_path: Path) -> None:
    asyncio.run(_run_queued_cancellation_case(tmp_path))
