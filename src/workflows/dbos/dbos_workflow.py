from __future__ import annotations

import asyncio
import logging
from typing import (
    Any,
    Tuple,
    cast,
)
import uuid
from typing_extensions import override

from llama_index_instrumentation import get_dispatcher

from workflows.context import Context
from .dbos_context import DBOSContext
from workflows.resource import ResourceManager
from workflows.workflow import Workflow
from workflows.decorators import StepConfig, step
from workflows.errors import (
    WorkflowConfigurationError,
    WorkflowDone,
    WorkflowTimeoutError,
)
from workflows.events import (
    StartEvent,
    StopEvent,
)
from workflows.handler import WorkflowHandler

from dbos import (
    DBOS,
    WorkflowHandle,
    error as dbos_error,
)

dispatcher = get_dispatcher(__name__)
logger = logging.getLogger()


class DBOSWorkflow(Workflow):
    """DBOS Workflow base class."""

    def __init__(
        self,
        timeout: float | None = 45.0,
        disable_validation: bool = False,
        verbose: bool = False,
        resource_manager: ResourceManager | None = None,
        num_concurrent_runs: int | None = None,
    ) -> None:
        """
        Initialize a DBOS workflow instance.

        Args:
            timeout (float | None): Max seconds to wait for completion. `None`
                disables the timeout.
            disable_validation (bool): Skip pre-run validation of the event graph
                (not recommended).
            verbose (bool): If True, print step activity.
            resource_manager (ResourceManager | None): Custom resource manager
                for dependency injection.
            num_concurrent_runs (int | None): Limit on concurrent `run()` calls.
        """
        super().__init__(
            timeout=timeout,
            disable_validation=disable_validation,
            verbose=verbose,
            resource_manager=resource_manager,
            num_concurrent_runs=num_concurrent_runs,
        )

        # TODO (Qian): add DBOS specific initialization if needed

    @override
    @step(num_workers=1)
    async def _done(self, ctx: Context, ev: StopEvent) -> None:
        """Tears down the whole workflow and stop execution."""
        if self._stop_event_class is StopEvent:
            ctx._retval = ev.result
        else:
            ctx._retval = ev

        ctx.write_event_to_stream(ev)

        # Signal all other workers to stop via DBOS durable notification
        dbos_ctx = cast(DBOSContext, ctx)
        await dbos_ctx.send_event_async("done")

        # Signal we want to stop the workflow. Since we're about to raise, delete
        # the reference to ctx explicitly to avoid it becoming dangling
        del ctx
        raise WorkflowDone

    @override
    def _start(
        self,
        ctx: Context | None = None,
    ) -> Tuple[Context, str]:
        """
        launches each step as a separate DBOS workflow.
        """
        run_id = str(uuid.uuid4())
        # Make sure it's a DBOSContext if provided
        if ctx is not None and not isinstance(ctx, DBOSContext):
            raise WorkflowConfigurationError(
                "Provided context is not a valid DBOSContext instance."
            )

        dbos_context = cast(DBOSContext, ctx) if ctx is not None else None
        if dbos_context is None:
            dbos_context = DBOSContext(self, context_name=f"dbos_ctx_{run_id}")
            self._contexts.add(dbos_context)
        else:
            # clean up the context from the previous run
            dbos_context._tasks = set()
            dbos_context._retval = None
            dbos_context._step_events_holding = None
            dbos_context._cancel_flag.clear()

        for name, step_func in self._get_steps().items():
            # DBOS shouldn't need these local queues and flags. Use durable notification instead.

            # At this point, step_func is guaranteed to have the `__step_config` attribute
            step_config: StepConfig = getattr(step_func, "__step_config")

            # Make the system step "_done" accept custom stop events
            if (
                name == "_done"
                and self._stop_event_class not in step_config.accepted_events
            ):
                step_config.accepted_events.append(self._stop_event_class)

            # Start DBOS workflows
            for _ in range(step_config.num_workers):
                dbos_context.add_step_worker(
                    name=name,
                    step=step_func,
                    config=step_config,
                    verbose=self._verbose,
                    run_id=run_id,
                    worker_id=str(uuid.uuid4()),
                    resource_manager=self._resource_manager,
                )

        # add dedicated cancel task
        dbos_context.add_cancel_worker()

        return dbos_context, run_id

    @override
    @dispatcher.span
    def run(
        self,
        ctx: Context | None = None,
        start_event: StartEvent | None = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        # Validate the workflow
        self._validate()

        # TODO(Qian): add a top level DBOS workflow somewhere to manage the steps? But how to make sure it guarantees determinism with asyncio.wait?
        async def _run_workflow(ctx: DBOSContext) -> None:
            if self._sem:
                await self._sem.acquire()

            try:
                if not ctx.is_running:
                    # Send the first event
                    start_event_instance = self._get_start_event_instance(
                        start_event, **kwargs
                    )
                    await ctx.send_event_async(start_event_instance)

                    # the context is now running
                    ctx.is_running = True

                def wait_for_completion(wf_handle: WorkflowHandle) -> None:
                    try:
                        wf_handle.get_result()
                    except dbos_error.DBOSAwaitedWorkflowCancelledError:
                        return

                wait_tasks: set[asyncio.Future] = set()
                for wf_handle in ctx._dbos_wf_handle:
                    wait_task = asyncio.get_event_loop().run_in_executor(
                        None, wait_for_completion, wf_handle
                    )
                    wait_tasks.add(wait_task)
                done, _ = await asyncio.wait(
                    wait_tasks,
                    timeout=self._timeout,
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                we_done = False
                exception_raised = None
                for task in done:
                    e = task.exception()
                    if type(e) is WorkflowDone:
                        we_done = True
                    elif e is not None:
                        exception_raised = e
                        break

                # Make sure all DBOS workflows are completed
                for wf_handle in ctx._dbos_wf_handle:
                    wf_id = wf_handle.get_workflow_id()
                    wf_handle_async = await DBOS.retrieve_workflow_async(wf_id)
                    try:
                        await wf_handle_async.get_result()
                    except Exception:
                        pass

                # the context is no longer running
                ctx.is_running = False

                if exception_raised:
                    # cancel the stream
                    ctx.write_event_to_stream(StopEvent())

                    raise exception_raised

                if not we_done:
                    # cancel the stream
                    ctx.write_event_to_stream(StopEvent())

                    msg = f"Operation timed out after {self._timeout} seconds"
                    raise WorkflowTimeoutError(msg)

                result.set_result(ctx._retval)
            except Exception as e:
                if not result.done():
                    result.set_exception(e)
            finally:
                if self._sem:
                    self._sem.release()
                await ctx.shutdown()

        # Start the machinery in a new Context or use the provided one
        started_ctx, run_id = self._start(ctx=ctx)
        run_task = asyncio.create_task(_run_workflow(cast(DBOSContext, started_ctx)))
        result = WorkflowHandler(ctx=started_ctx, run_id=run_id, run_task=run_task)
        return result
