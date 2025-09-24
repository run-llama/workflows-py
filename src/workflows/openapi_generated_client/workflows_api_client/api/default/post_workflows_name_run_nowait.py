from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.handler import Handler
from ...models.post_workflows_name_run_nowait_body import PostWorkflowsNameRunNowaitBody
from ...types import Response


def _get_kwargs(
    name: str,
    *,
    body: PostWorkflowsNameRunNowaitBody,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": f"/workflows/{name}/run-nowait",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, Handler]]:
    if response.status_code == 200:
        response_200 = Handler.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = cast(Any, None)
        return response_400

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, Handler]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PostWorkflowsNameRunNowaitBody,
) -> Response[Union[Any, Handler]]:
    """Run workflow (no-wait)

     Starts the specified workflow asynchronously and returns a handler identifier
    which can be used to query results or stream events.

    Args:
        name (str):
        body (PostWorkflowsNameRunNowaitBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, Handler]]
    """

    kwargs = _get_kwargs(
        name=name,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PostWorkflowsNameRunNowaitBody,
) -> Optional[Union[Any, Handler]]:
    """Run workflow (no-wait)

     Starts the specified workflow asynchronously and returns a handler identifier
    which can be used to query results or stream events.

    Args:
        name (str):
        body (PostWorkflowsNameRunNowaitBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, Handler]
    """

    return sync_detailed(
        name=name,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PostWorkflowsNameRunNowaitBody,
) -> Response[Union[Any, Handler]]:
    """Run workflow (no-wait)

     Starts the specified workflow asynchronously and returns a handler identifier
    which can be used to query results or stream events.

    Args:
        name (str):
        body (PostWorkflowsNameRunNowaitBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, Handler]]
    """

    kwargs = _get_kwargs(
        name=name,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    name: str,
    *,
    client: Union[AuthenticatedClient, Client],
    body: PostWorkflowsNameRunNowaitBody,
) -> Optional[Union[Any, Handler]]:
    """Run workflow (no-wait)

     Starts the specified workflow asynchronously and returns a handler identifier
    which can be used to query results or stream events.

    Args:
        name (str):
        body (PostWorkflowsNameRunNowaitBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, Handler]
    """

    return (
        await asyncio_detailed(
            name=name,
            client=client,
            body=body,
        )
    ).parsed
