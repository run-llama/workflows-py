# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from llama_agents.server._runtime.idle_release_runtime import IdleReleaseDecorator
from llama_agents.server._runtime.persistence_runtime import PersistenceDecorator
from llama_agents.server._runtime.server_runtime import ServerRuntimeDecorator
from starlette.middleware import Middleware
from workflows import Workflow
from workflows.events import Event
from workflows.plugins.basic import basic_runtime
from workflows.runtime.types.plugin import Runtime

from ._api import _WorkflowAPI
from ._service import _WorkflowService
from ._store.abstract_workflow_store import AbstractWorkflowStore
from ._store.memory_workflow_store import MemoryWorkflowStore

logger = logging.getLogger()


class WorkflowServer:
    def __init__(
        self,
        *,
        middleware: list[Middleware] | None = None,
        workflow_store: AbstractWorkflowStore | None = None,
        # retry/backoff seconds for persisting the handler state in the store after failures. Configurable mainly for testing.
        persistence_backoff: list[float] = [0.5, 3],
        runtime: Runtime | None = None,
    ):
        self._workflow_store = (
            workflow_store if workflow_store is not None else MemoryWorkflowStore()
        )
        inner: Runtime = (
            runtime
            if runtime is not None
            else IdleReleaseDecorator(
                PersistenceDecorator(basic_runtime, store=self._workflow_store),
                store=self._workflow_store,
            )
        )
        self._runtime: ServerRuntimeDecorator = ServerRuntimeDecorator(
            inner,
            store=self._workflow_store,
            persistence_backoff=list(persistence_backoff),
        )
        self._service = _WorkflowService(
            runtime=self._runtime, store=self._workflow_store
        )

        self._api = _WorkflowAPI(self._service, middleware=middleware)
        self.app = self._api.app

    # ------------------------------------------------------------------
    # Workflow registration
    # ------------------------------------------------------------------

    def add_workflow(
        self,
        name: str,
        workflow: Workflow,
        additional_events: list[type[Event]] | None = None,
    ) -> None:
        workflow._switch_workflow_name(name)
        workflow._switch_runtime(self._runtime)

        if additional_events is not None:
            self._api.register_additional_events(name, additional_events)

    def get_workflows(self) -> dict[str, Workflow]:
        """Return registered workflows as a dict by name. Only available after start()."""
        return {
            n: wf
            for n in self._service.get_workflow_names()
            if (wf := self._service.get_workflow(n)) is not None
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> WorkflowServer:
        """Resumes previously running workflows, if they were not complete at last shutdown.

        Idle workflows are not resumed - they remain released and will be
        loaded on-demand when events arrive for them.
        """
        await self._service.start()
        return self

    @asynccontextmanager
    async def contextmanager(self) -> AsyncGenerator[WorkflowServer, None]:
        """Use this server as a context manager to start and stop it"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def stop(self) -> None:
        await self._service.stop()

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------

    async def serve(
        self,
        host: str = "localhost",
        port: int = 80,
        uvicorn_config: dict[str, Any] | None = None,
    ) -> None:
        """Run the server."""
        uvicorn_config = uvicorn_config or {}

        config = uvicorn.Config(self.app, host=host, port=port, **uvicorn_config)
        server = uvicorn.Server(config)
        logger.info(
            f"Starting Workflow server at http://{host}:{port}{uvicorn_config.get('root_path', '/')}"
        )

        await server.serve()

    def openapi_schema(self) -> dict:
        return self._api.openapi_schema()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate OpenAPI schema")
    parser.add_argument(
        "--output", type=str, default="openapi.json", help="Output file path"
    )
    args = parser.parse_args()

    server = WorkflowServer()
    dict_schema = server.openapi_schema()
    with open(args.output, "w") as f:
        json.dump(dict_schema, indent=2, fp=f)
    print(f"OpenAPI schema written to {args.output}")  # noqa: T201
