# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from workflows import Workflow
from workflows.events import Event

from ._api import _WorkflowAPI
from ._service import _WorkflowService
from .abstract_workflow_store import AbstractWorkflowStore

logger = logging.getLogger()


class WorkflowServer:
    def __init__(
        self,
        *,
        middleware: list[Middleware] | None = None,
        workflow_store: AbstractWorkflowStore | None = None,
        # retry/backoff seconds for persisting the handler state in the store after failures. Configurable mainly for testing.
        persistence_backoff: list[float] = [0.5, 3],
        # Release idle workflows from memory after this timeout (None = disabled)
        idle_release_timeout: timedelta | None = timedelta(seconds=10),
    ):
        self._service = _WorkflowService(
            workflow_store=workflow_store,
            persistence_backoff=persistence_backoff,
            idle_release_timeout=idle_release_timeout,
        )
        self._api = _WorkflowAPI(self._service)
        self._assets_path = Path(__file__).parent / "static"

        self._middleware = middleware or [
            Middleware(
                CORSMiddleware,  # type: ignore[arg-type]
                # regex echoes the origin header back, which some browsers require (rather than "*") when credentials are required
                allow_origin_regex=".*",
                allow_methods=["*"],
                allow_headers=["*"],
                allow_credentials=True,
            )
        ]

        self._routes = [
            Route("/workflows", self._api._list_workflows, methods=["GET"]),
            Route(
                "/workflows/{name}/run",
                self._api._run_workflow,
                methods=["POST"],
            ),
            Route(
                "/workflows/{name}/run-nowait",
                self._api._run_workflow_nowait,
                methods=["POST"],
            ),
            Route(
                "/workflows/{name}/schema",
                self._api._get_events_schema,
                methods=["GET"],
            ),
            Route(
                "/results/{handler_id}",
                self._api._get_workflow_result,
                methods=["GET"],
            ),
            Route(
                "/events/{handler_id}",
                self._api._stream_events,
                methods=["GET"],
            ),
            Route(
                "/events/{handler_id}",
                self._api._post_event,
                methods=["POST"],
            ),
            Route("/health", self._api._health_check, methods=["GET"]),
            Route("/handlers", self._api._get_handlers, methods=["GET"]),
            Route(
                "/handlers/{handler_id}",
                self._api._get_workflow_handler,
                methods=["GET"],
            ),
            Route(
                "/handlers/{handler_id}/cancel",
                self._api._cancel_handler,
                methods=["POST"],
            ),
            Route(
                "/workflows/{name}/representation",
                self._api._get_workflow_representation,
                methods=["GET"],
            ),
            Route(
                "/workflows/{name}/events",
                self._api._list_workflow_events,
                methods=["GET"],
            ),
        ]

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
            async with self.contextmanager():
                yield

        self.app = Starlette(
            routes=self._routes,
            middleware=self._middleware,
            lifespan=lifespan,
        )
        # Serve the UI as static files
        self.app.mount(
            "/", app=StaticFiles(directory=self._assets_path, html=True), name="ui"
        )

    def add_workflow(
        self,
        name: str,
        workflow: Workflow,
        additional_events: list[type[Event]] | None = None,
    ) -> None:
        self._service.add_workflow(name, workflow, additional_events)

    async def start(self) -> "WorkflowServer":
        """Resumes previously running workflows, if they were not complete at last shutdown.

        Idle workflows are not resumed - they remain released and will be
        loaded on-demand when events arrive for them.
        """
        await self._service.start()
        return self

    @asynccontextmanager
    async def contextmanager(self) -> AsyncGenerator["WorkflowServer", None]:
        """Use this server as a context manager to start and stop it"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def stop(self) -> None:
        await self._service.stop()

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
        return self._api.openapi_schema(self.app)


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
