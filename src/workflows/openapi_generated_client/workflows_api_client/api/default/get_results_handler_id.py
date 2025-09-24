from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.handler import Handler
from ...types import Response


def _get_kwargs(
    handler_id: str,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/results/{handler_id}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, Handler, str]]:
    if response.status_code == 200:
        response_200 = Handler.from_dict(response.json())

        return response_200

    if response.status_code == 202:
        response_202 = Handler.from_dict(response.json())

        return response_202

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if response.status_code == 500:
        response_500 = response.text
        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, Handler, str]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[Any, Handler, str]]:
    """Get workflow result

     Returns the final result of an asynchronously started workflow, if available

    Args:
        handler_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, Handler, str]]
    """

    kwargs = _get_kwargs(
        handler_id=handler_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[Any, Handler, str]]:
    """Get workflow result

     Returns the final result of an asynchronously started workflow, if available

    Args:
        handler_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, Handler, str]
    """

    return sync_detailed(
        handler_id=handler_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[Any, Handler, str]]:
    """Get workflow result

     Returns the final result of an asynchronously started workflow, if available

    Args:
        handler_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, Handler, str]]
    """

    kwargs = _get_kwargs(
        handler_id=handler_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    handler_id: str,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[Any, Handler, str]]:
    """Get workflow result

     Returns the final result of an asynchronously started workflow, if available

    Args:
        handler_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, Handler, str]
    """

    return (
        await asyncio_detailed(
            handler_id=handler_id,
            client=client,
        )
    ).parsed
