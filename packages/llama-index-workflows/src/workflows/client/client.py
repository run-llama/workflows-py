# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    overload,
)

import httpx

from workflows import Context
from workflows.events import Event, StartEvent
from workflows.protocol import (
    CancelHandlerResponse,
    HandlerData,
    HandlersListResponse,
    HealthResponse,
    SendEventResponse,
    Status,
    WorkflowsListResponse,
)
from workflows.protocol.serializable_events import (
    EventEnvelope,
    EventEnvelopeWithMetadata,
)


def _raise_for_status_with_body(response: httpx.Response) -> None:
    """
    Raise an HTTPStatusError with the first 200 characters of the response body
    for 400 and 500 level errors.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if 400 <= e.response.status_code < 600:
            body_preview = e.response.text[:200]
            method = e.request.method
            url = e.request.url
            status_code = e.response.status_code
            raise httpx.HTTPStatusError(
                f"{status_code} {e.response.reason_phrase} for {method} {url}. Response: {body_preview}",
                request=e.request,
                response=e.response,
            ) from e
        raise


class WorkflowClient:
    @overload
    def __init__(self, *, httpx_client: httpx.AsyncClient): ...
    @overload
    def __init__(
        self,
        *,
        base_url: str,
    ): ...

    def __init__(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
    ):
        if httpx_client is None and base_url is None:
            raise ValueError("Either httpx_client or base_url must be provided")
        if httpx_client is not None and base_url is not None:
            raise ValueError("Only one of httpx_client or base_url must be provided")
        self.httpx_client = httpx_client
        self.base_url = base_url

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self.httpx_client:
            yield self.httpx_client
        else:
            async with httpx.AsyncClient(base_url=self.base_url or "") as client:
                yield client

    async def is_healthy(self) -> HealthResponse:
        """
        Check whether the workflow server is helathy or not

        Returns:
            HealthResponse: health response from the workflow
        """
        async with self._get_client() as client:
            response = await client.get("/health")
            _raise_for_status_with_body(response)
            return HealthResponse.model_validate(response.json())

    async def list_workflows(self) -> WorkflowsListResponse:
        """
        List workflows

        Returns:
            WorkflowsListResponse: List of workflow names available through the server.
        """
        async with self._get_client() as client:
            response = await client.get("/workflows")

            _raise_for_status_with_body(response)

            return WorkflowsListResponse.model_validate(response.json())

    async def run_workflow(
        self,
        workflow_name: str,
        handler_id: str | None = None,
        start_event: StartEvent | dict[str, Any] | None = None,
        context: Context | dict[str, Any] | None = None,
    ) -> HandlerData:
        """
        Run the workflow and wait until completion.

        Args:
            start_event (Union[StartEvent, dict[str, Any], None]): start event class or dictionary representation (optional, defaults to None and get passed as an empty dictionary if not provided).
            context: Context or serialized representation of it (optional, defaults to None if not provided)
            handler_id (Optional[str]): Workflow handler identifier to continue from a previous completed run.

        Returns:
            HandlerData: Data representing the handler running the workflow (including result and metadata)
        """
        if start_event is not None:
            try:
                start_event = _serialize_event(start_event, bare=True)
            except Exception as e:
                raise ValueError(
                    f"Impossible to serialize the start event because of: {e}"
                )
        if isinstance(context, Context):
            try:
                context = context.to_dict()
            except Exception as e:
                raise ValueError(f"Impossible to serialize the context because of: {e}")
        request_body = {
            "start_event": start_event or "",
            "context": context or {},
        }
        if handler_id:
            request_body["handler_id"] = handler_id
        async with self._get_client() as client:
            response = await client.post(
                f"/workflows/{workflow_name}/run", json=request_body
            )

            _raise_for_status_with_body(response)

            return HandlerData.model_validate(response.json())

    async def run_workflow_nowait(
        self,
        workflow_name: str,
        handler_id: str | None = None,
        start_event: StartEvent | dict[str, Any] | None = None,
        context: Context | dict[str, Any] | None = None,
    ) -> HandlerData:
        """
        Run the workflow in the background.

        Args:
            start_event (Union[StartEvent, dict[str, Any], None]): start event class or dictionary representation (optional, defaults to None and get passed as an empty dictionary if not provided).
            context: Context or serialized representation of it (optional, defaults to None if not provided)
            handler_id (Optional[str]): Workflow handler identifier to continue from a previous completed run.

        Returns:
            HandlerData: data representing the handler running the workflow.
        """
        if start_event is not None:
            try:
                start_event = _serialize_event(start_event)
            except Exception as e:
                raise ValueError(
                    f"Impossible to serialize the start event because of: {e}"
                )
        if isinstance(context, Context):
            try:
                context = context.to_dict()
            except Exception as e:
                raise ValueError(f"Impossible to serialize the context because of: {e}")
        request_body: dict[str, Any] = {
            "start_event": start_event or _serialize_event(StartEvent()),
            "context": context or {},
        }
        if handler_id:
            request_body["handler_id"] = handler_id
        async with self._get_client() as client:
            response = await client.post(
                f"/workflows/{workflow_name}/run-nowait", json=request_body
            )

            _raise_for_status_with_body(response)

            return HandlerData.model_validate(response.json())

    async def get_workflow_events(
        self,
        handler_id: str,
        include_internal_events: bool = False,
        lock_timeout: float = 1,
    ) -> AsyncGenerator[EventEnvelopeWithMetadata, None]:
        """
        Stream events as they are produced by the workflow.

        Args:
            handler_id (str): ID of the handler running the workflow
            include_internal_events (bool): Include internal workflow events. Defaults to False.
            lock_timeout (float): Timeout (in seconds) for acquiring the lock to iterate over the events.

        Returns:
            AsyncGenerator[EventEnvelopeWithMetadata, None]: Generator for the events that are streamed as instances of `EventEnvelopeWithMetadata`.
        """
        incl_inter = "true" if include_internal_events else "false"
        url = f"/events/{handler_id}"

        async with self._get_client() as client:
            try:
                async with client.stream(
                    "GET",
                    url,
                    params={
                        "sse": "false",
                        "include_internal": incl_inter,
                        "acquire_timeout": lock_timeout,
                    },
                    headers={"Connection": "keep-alive"},
                    timeout=None,
                ) as response:
                    # Handle different response codes
                    if response.status_code == 404:
                        raise ValueError("Handler not found")
                    elif response.status_code == 204:
                        # Handler completed, no more events
                        return

                    _raise_for_status_with_body(response)

                    async for line in response.aiter_lines():
                        if line.strip():  # Skip empty lines
                            event = EventEnvelopeWithMetadata.model_validate_json(line)
                            yield event

            except httpx.TimeoutException:
                raise TimeoutError(
                    f"Timeout waiting for events from handler {handler_id}"
                )
            except httpx.RequestError as e:
                raise ConnectionError(f"Failed to connect to event stream: {e}")

    async def send_event(
        self,
        handler_id: str,
        event: Event | dict[str, Any],
        step: str | None = None,
    ) -> SendEventResponse:
        """
        Send an event to the workflow.

        Args:
            handler_id (str): ID of the handler of the running workflow to send the event to
            event (Event | dict[str, Any] | str): Event to send, represented as an Event object, a dictionary or a serialized string.
            step (Optional[str]): Step to send the event to (optional, defaults to None)

        Returns:
            SendEventResponse: Confirmation of the send operation
        """
        try:
            serialized_event: dict[str, Any] = _serialize_event(event)
        except Exception as e:
            raise ValueError(f"Error while serializing the provided event: {e}")
        request_body: dict[str, Any] = {"event": serialized_event}
        if step:
            request_body["step"] = step
        async with self._get_client() as client:
            response = await client.post(f"/events/{handler_id}", json=request_body)
            _raise_for_status_with_body(response)

            return SendEventResponse.model_validate(response.json())

    async def get_result(self, handler_id: str) -> HandlerData:
        """
        Deprecated. Use get_handler instead.
        """
        return await self.get_handler(handler_id)

    async def get_handlers(
        self,
        status: list[Status] | None = None,
        workflow_name: list[str] | None = None,
    ) -> HandlersListResponse:
        """
        Get all the workflow handlers.
        Args:
            status (list[Status] | None): List of statuses (e.g. "running", "completed", etc. ) to filter by. Defaults to None.
            workflow_name (list[str] | None): List of workflow names to filter by. Defaults to None.
        Returns:
            HandlersListResponse: List of workflow handlers.
        """
        async with self._get_client() as client:
            response = await client.get(
                "/handlers",
                params={
                    "status": status,
                    "workflow_name": workflow_name,
                },
            )
            _raise_for_status_with_body(response)

            return HandlersListResponse.model_validate(response.json())

    async def get_handler(self, handler_id: str) -> HandlerData:
        """
        Get a single workflow handler by identifier.

        Args:
            handler_id (str): ID of the handler associated with the workflow run

        Returns:
            HandlerData: Handler metadata persisted by the server.
        """
        async with self._get_client() as client:
            response = await client.get(f"/handlers/{handler_id}")
            _raise_for_status_with_body(response)

            return HandlerData.model_validate(response.json())

    async def cancel_handler(
        self, handler_id: str, purge: bool = False
    ) -> CancelHandlerResponse:
        """
        Stop and cancel a workflow run.

        Args:
            handler_id (str): ID of the handler associated with the workflow run
            purge (bool): Whether or not to delete the run also from the persistent storage. Defaults to false
        """
        async with self._get_client() as client:
            response = await client.post(
                f"/handlers/{handler_id}/cancel",
                params={"purge": "true" if purge else "false"},
            )
            _raise_for_status_with_body(response)

            return CancelHandlerResponse.model_validate(response.json())


def _serialize_event(
    event: Event | dict[str, Any], bare: bool = False
) -> dict[str, Any]:
    if isinstance(event, dict):
        return event  # assumes you know what you are doing. In many cases this needs to be a dict that contains type metadata and the value
    return (
        event.model_dump()
        if bare
        else EventEnvelope.from_event(event=event).model_dump()
    )
