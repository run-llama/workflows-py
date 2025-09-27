import httpx
import time
import json
import inspect

from typing import Literal, Any, Union, Callable, AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from logging import getLogger
from workflows.events import StartEvent, Event
from workflows import Context


logger = getLogger(__name__)


class WorkflowClient:
    def __init__(
        self,
        protocol: Literal["http", "https"] | None = None,
        host: str | None = None,
        port: int | None = None,
        timeout: int | None = None,
    ):
        # TODO: middleware-related logic
        self.protocol = protocol or "http"
        self.host = host or "localhost"
        self.port = port or 8000
        self.timeout = timeout or 600
        # TODO: add some basic TLS/verification and auth features

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator:
        async with httpx.AsyncClient(
            base_url=self.protocol + "://" + self.host + ":" + str(self.port),
            timeout=self.timeout,
        ) as client:
            yield client

    async def is_healthy(self) -> bool:
        """
        Check whether the workflow server is helathy or not

        Returns:
            bool: True if the workflow server is healthy, false if not
        """
        async with self._get_client() as client:
            response = await client.get("/health")
        if response.status_code == 200:
            return response.json().get("status", "") == "healthy"
        return False

    async def ping(self) -> float:
        """
        Ping the workflow and get the latency in milliseconds

        Returns:
            float: latency in milliseconds
        """
        async with self._get_client() as client:
            start = time.time()
            response = await client.get("/health")
            if response.status_code == 200:
                end = time.time()
                return (end - start) * 1000
            else:
                raise httpx.ConnectError(
                    f"Failed to establish a connection with server running on: {self.protocol}://{self.host}:{self.port}"
                )

    async def list_workflows(self) -> list[str]:
        """
        List workflows

        Returns:
            list: List of workflow names available through the server.
        """
        async with self._get_client() as client:
            response = await client.get("/workflows")

            response.raise_for_status()

            return response.json()["workflows"]

    async def run_workflow(
        self,
        workflow_name: str,
        start_event: Union[StartEvent, dict[str, Any], None] = None,
        context: Union[Context, dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the workflow and wait until completion.

        Args:
            start_event (Union[StartEvent, dict[str, Any], None]): start event class or dictionary representation (optional, defaults to None and get passed as an empty dictionary if not provided).
            context: Context or serialized representation of it (optional, defaults to None if not provided)
            **kwargs: Any number of keyword arguments that would be passed on as additional keyword arguments to the workflow.

        Returns:
            Any: Result of the workflow
        """
        if isinstance(start_event, StartEvent):
            try:
                start_event = start_event.model_dump()
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
            "start_event": start_event or {},
            "context": context or {},
            "additional_kwargs": kwargs,
        }
        async with self._get_client() as client:
            response = await client.post(
                f"/workflows/{workflow_name}/run", json=request_body
            )

            response.raise_for_status()

            return response.json()["result"]

    async def run_workflow_nowait(
        self,
        workflow_name: str,
        start_event: Union[StartEvent, dict[str, Any], None] = None,
        context: Union[Context, dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the workflow in the background.

        Args:
            start_event (Union[StartEvent, dict[str, Any], None]): start event class or dictionary representation (optional, defaults to None and get passed as an empty dictionary if not provided).
            context: Context or serialized representation of it (optional, defaults to None if not provided)
            **kwargs: Any number of keyword arguments that would be passed on as additional keyword arguments to the workflow.

        Returns:
            str: ID of the handler running the workflow
        """
        if isinstance(start_event, StartEvent):
            try:
                start_event = start_event.model_dump()
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
            "start_event": start_event or {},
            "context": context or {},
            "additional_kwargs": kwargs,
        }
        async with self._get_client() as client:
            response = await client.post(
                f"/workflows/{workflow_name}/run-nowait", json=request_body
            )

            response.raise_for_status()

            return response.json()["handler_id"]

    async def _stream_events_sse(
        self,
        handler_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream events using Server-Sent Events format
        """
        url = f"/events/{handler_id}?sse=true"

        async with self._get_client() as client:
            try:
                async with client.stream(
                    "GET",
                    url,
                ) as response:
                    # Handle different response codes
                    if response.status_code == 404:
                        raise ValueError("Handler not found")
                    elif response.status_code == 204:
                        # Handler completed, no more events
                        return  # type: ignore

                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            # Extract JSON from SSE data line
                            json_data = line[6:]  # Remove 'data: ' prefix
                            if json_data.strip():  # Skip empty data lines
                                try:
                                    event = json.loads(json_data.replace("\n", ""))
                                    yield event.get("value", {})
                                except json.JSONDecodeError as e:
                                    print(
                                        f"Failed to parse JSON: {e}, data: {json_data}"
                                    )
                                    continue

            except httpx.TimeoutException:
                raise TimeoutError(
                    f"Timeout waiting for events from handler {handler_id}"
                )
            except httpx.RequestError as e:
                raise ConnectionError(f"Failed to connect to event stream: {e}")

    async def _stream_events_ndjson(
        self,
        handler_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream events using newline-delimited JSON format
        """
        url = f"/events/{handler_id}?sse=false"

        async with self._get_client() as client:
            try:
                async with client.stream("GET", url) as response:
                    # Handle different response codes
                    if response.status_code == 404:
                        raise ValueError("Handler not found")
                    elif response.status_code == 204:
                        # Handler completed, no more events
                        return

                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip():  # Skip empty lines
                            try:
                                event = json.loads(line.replace("\n", ""))
                                yield event.get("value", {})
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse JSON: {e}, data: {line}")
                                continue

            except httpx.TimeoutException:
                raise TimeoutError(
                    f"Timeout waiting for events from handler {handler_id}"
                )
            except httpx.RequestError as e:
                raise ConnectionError(f"Failed to connect to event stream: {e}")

    async def stream_events(
        self,
        handler_id: str,
        event_callback: Callable[[dict[str, Any]], Any] | None = None,
        sse: bool = True,
    ) -> None:
        """
        Stream events from a running handler.

        Args:
            handler_id (str): ID of the handler streaming the events
            event_callback (Callable[[dict[str, Any]], Any]): Function to call when an event is received from the stream (optional, defaults to None)
            sse (bool): Whether to enable server-sent events or not

        Returns:
            None
        """
        callback = event_callback or (
            lambda event: logger.info(f"Processing data: {event}")
        )
        is_async = inspect.iscoroutinefunction(callback)
        if sse:
            async for event in self._stream_events_sse(handler_id):
                if is_async:
                    await callback(event)  # type: ignore
                else:
                    callback(event)
        else:
            async for event in self._stream_events_ndjson(handler_id):
                if is_async:
                    await callback(event)  # type: ignore
                else:
                    callback(event)
        return None

    async def send_event(
        self,
        handler_id: str,
        event: Event | dict[str, Any] | str,
        step: str | None = None,
    ) -> bool:
        """
        Send an event to the workflow.

        Args:
            handler_id (str): ID of the handler of the running workflow to send the event to
            event (Event | dict[str, Any] | str): Event to send, represented as an Event object, a dictionary or a serialized string.
            step (str | None): Step to send the event to (optional, defaults to None)

        Returns:
            bool: Success status of the send operation
        """
        if isinstance(event, Event):
            try:
                event = event.model_dump_json()
            except Exception as e:
                raise ValueError(f"Error while serializing the provided event: {e}")
        elif isinstance(event, dict):
            try:
                event = json.dumps(event)
            except Exception as e:
                raise ValueError(f"Error while serializing the provided event: {e}")
        request_body = {"event": event}
        if step:
            request_body.update({"step": step})
        async with self._get_client() as client:
            response = await client.post(f"/events/{handler_id}", json=request_body)
            response.raise_for_status()

            return response.json()["status"] == "sent"

    async def get_result(self, handler_id: str) -> Any:
        """
        Get the result of the workflow associated with the specified handler ID.

        Args:
            handler_id (str): ID of the handler running the workflow

        Returns:
            Any: Result of the workflow
        """
        async with self._get_client() as client:
            response = await client.get(f"/results/{handler_id}")
            response.raise_for_status()

            if response.status_code == 202:
                return

            return response.json()["result"]
