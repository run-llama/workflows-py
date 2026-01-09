# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Literal,
    Type,
    TypeVar,
    cast,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

T = TypeVar("T")
B = TypeVar("B", bound=BaseModel)


class _Resource(Generic[T]):
    """Internal wrapper for resource factories.

    Wraps sync/async factories and records metadata such as the qualified name
    and cache behavior.
    """

    def __init__(self, factory: Callable[..., T | Awaitable[T]], cache: bool) -> None:
        self._factory = factory
        self._is_async = inspect.iscoroutinefunction(factory)
        self.type: Literal["function"] = "function"
        self.name = getattr(factory, "__qualname__", type(factory).__name__)
        self.cache = cache

    async def call(self) -> T:
        """Invoke the underlying factory, awaiting if necessary."""
        if self._is_async:
            result = await cast(Callable[..., Awaitable[T]], self._factory)()
        else:
            result = cast(Callable[..., T], self._factory)()
        return result


class _ConfiguredResource(Generic[B]):
    """
    Internal wrapper for a pydantic-based resource whose configuration can be read from a JSON file.
    """

    def __init__(
        self, config_file: str, cache: bool, cls_factory: Type[B] | None = None
    ) -> None:
        if not Path(config_file).is_file():
            raise FileNotFoundError(f"No such file: {config_file}")
        if Path(config_file).suffix != ".json":
            raise ValueError(
                "Only JSON files can be used to load Pydantic-based resources."
            )
        self.config_file = config_file
        self.cls_factory = cls_factory
        self.type: Literal["configured"] = "configured"
        self.cache = cache
        self.name = config_file  # for caching purposes

    # make async for compatibility with _Resource
    async def call(self) -> B:
        with open(self.config_file, "r") as f:
            data = json.load(f)
        # let validation error bubble up
        if self.cls_factory is not None:
            return self.cls_factory.model_validate(data)
        raise ValueError(
            "Class factory should be set to a BaseModel subclass before calling"
        )


class ResourceDefinition(BaseModel):
    """Definition for a resource injection requested by a step signature.

    Attributes:
        name (str): Parameter name in the step function.
        resource (_Resource): Factory wrapper used by the manager to produce the dependency.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    resource: _ConfiguredResource | _Resource
    type_annotation: Any = None


def Resource(
    factory: Callable[..., T] | None = None,
    config_file: str | None = None,
    cache: bool = True,
) -> _ConfiguredResource | _Resource:
    """Declare a resource to inject into step functions.

    Args:
        factory (Callable[..., T] | None): Function returning the resource instance. May be async.
        config_file (str | None): Configuration file where to load the resource from. Only applicable to resources associated with a Pydantic class.
        cache (bool): If True, reuse the produced resource across steps. Defaults to True.

    Returns:
        _Resource[T] | _ConfiguredResource[BaseModel]: A resource descriptor to be used in `typing.Annotated`.

    Examples:

        With function factories:

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

        With Pydantic models:

        ```python
        import json

        from typing import Annotated
        from pydantic import BaseModel
        from workflows.resource import Resource

        class Memory(BaseModel):
            messages: list[str]

        memory = {"messages": ["hello", "how are you?"]}
        with open("config.json", "w") as f:
            json.dump(memory, f, indent=2)

        class MyWorkflow(Workflow):
            @step
            async def step_with_memory(self,
                ev: StartEvent,
                memory: Annotated[Memory, Resource(config_file="config.json")]
            ) -> StopEvent:
                ...
        ```
    """
    if factory is not None:
        return _Resource(factory, cache)
    elif config_file is not None:
        return _ConfiguredResource(config_file=config_file, cache=cache)
    else:
        raise ValueError(
            "At least one between `factory` and `config_file` has to be provided"
        )


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

    async def get(self, resource: _Resource | _ConfiguredResource) -> Any:
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
