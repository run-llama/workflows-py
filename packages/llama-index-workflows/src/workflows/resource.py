# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    TypeVar,
    cast,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

T = TypeVar("T")


class _Resource(Generic[T]):
    """Internal wrapper for resource factories.

    Wraps sync/async factories and records metadata such as the qualified name
    and cache behavior.
    """

    def __init__(self, factory: Callable[..., T | Awaitable[T]], cache: bool) -> None:
        self._factory = factory
        self._is_async = inspect.iscoroutinefunction(factory)
        self.name = factory.__qualname__
        self.cache = cache

    async def call(self) -> T:
        """Invoke the underlying factory, awaiting if necessary."""
        if self._is_async:
            result = await cast(Callable[..., Awaitable[T]], self._factory)()
        else:
            result = cast(Callable[..., T], self._factory)()
        return result


class ResourceDefinition(BaseModel):
    """Definition for a resource injection requested by a step signature.

    Attributes:
        name (str): Parameter name in the step function.
        resource (_Resource): Factory wrapper used by the manager to produce the dependency.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    resource: _Resource


def Resource(factory: Callable[..., T], cache: bool = True) -> _Resource[T]:
    """Declare a resource to inject into step functions.

    Args:
        factory (Callable[..., T]): Function returning the resource instance. May be async.
        cache (bool): If True, reuse the produced resource across steps. Defaults to True.

    Returns:
        _Resource[T]: A resource descriptor to be used in `typing.Annotated`.

    Examples:
        ```python
        from typing import Annotated
        from workflows.resource import Resource

        def get_memory(**kwargs) -> Memory:
            return Memory.from_defaults("user123", token_limit=60000)

        class MyWorkflow(Workflow):
            @step
            async def first(
                self,
                ev: StartEvent,
                memory: Annotated[Memory, Resource(get_memory)],
            ) -> StopEvent:
                await memory.aput(...)
                return StopEvent(result="ok")
        ```
    """
    return _Resource(factory, cache)


class ResourceManager:
    """Manage resource lifecycles and caching across workflow steps.

    Methods:
        set: Manually set a resource by name.
        get: Produce or retrieve a resource via its descriptor.
        get_all: Return the internal name->resource map.
    """

    def __init__(self) -> None:
        self.resources: dict[str, Any] = {}

    async def set(self, name: str, val: Any) -> None:
        """Register a resource instance under a name."""
        self.resources.update({name: val})

    async def get(self, resource: _Resource) -> Any:
        """Return a resource instance, honoring cache settings."""
        if not resource.cache:
            val = await resource.call()
        elif resource.cache and not self.resources.get(resource.name, None):
            val = await resource.call()
            await self.set(resource.name, val)
        else:
            val = self.resources.get(resource.name)
        return val

    def get_all(self) -> dict[str, Any]:
        """Return all materialized resources."""
        return self.resources
