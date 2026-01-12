# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
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
        self.name = getattr(factory, "__qualname__", type(factory).__name__)
        self.cache = cache
        self.resource_configs: Optional[dict[str, BaseModel]] = None  # noqa: UP045

    def prepare_resource_configs(self) -> None:
        if self.resource_configs is None:
            params = inspect.signature(self._factory).parameters
            resource_configs: dict[str, BaseModel] = {}
            if len(params) > 0:
                for param in params.values():
                    if get_origin(param.annotation) is Annotated:
                        args = get_args(param.annotation)
                        if len(args) == 2 and isinstance(args[1], _ResourceConfig):
                            resource_config = args[1]
                            resource_config.cls_factory = args[0]
                            value = resource_config.call()
                            resource_configs.update({param.name: value})
            self.resource_configs = resource_configs
        return None

    async def call(self) -> T:
        """Invoke the underlying factory, awaiting if necessary."""
        self.prepare_resource_configs()
        args = cast(dict[str, BaseModel], self.resource_configs)
        if self._is_async:
            result = await cast(Callable[..., Awaitable[T]], self._factory)(**args)
        else:
            result = cast(Callable[..., T], self._factory)(**args)
        return result


@functools.lru_cache(maxsize=1)
def _get_resource_config_data(
    config_file: str,
    path_selector: str | None,
) -> dict[str, Any]:
    with open(config_file, "r") as f:
        data = json.load(f)
    if path_selector is not None:
        keys = path_selector.split(".")
        val: dict[str, Any] = data
        cumulative_path = ""
        for key in keys:
            cumulative_path += key + "."
            got = cast(Optional[dict[str, Any]], val.get(key))  # noqa: UP045
            if not isinstance(got, dict):
                raise ValueError(
                    f"Expected dictionary for configuration from {config_file} at path {cumulative_path.strip('.')}, got: {type(got)}"
                )
            val = got
        return val
    return data


class _ResourceConfig(Generic[B]):
    """
    Internal wrapper for a pydantic-based resource whose configuration can be read from a JSON file.
    """

    def __init__(
        self,
        config_file: str,
        path_selector: str | None,
        cls_factory: Type[B] | None = None,
    ) -> None:
        if not Path(config_file).is_file():
            raise FileNotFoundError(f"No such file: {config_file}")
        if Path(config_file).suffix != ".json":
            raise ValueError(
                "Only JSON files can be used to load Pydantic-based resources."
            )
        self.config_file = config_file
        self.path_selector = path_selector
        self.cls_factory = cls_factory

    @property
    def name(self) -> str:
        if self.path_selector is not None:
            return self.config_file + "." + self.path_selector
        return self.config_file

    # make async for compatibility with _Resource
    def call(self) -> B:
        sel_data = _get_resource_config_data(
            config_file=self.config_file, path_selector=self.path_selector
        )
        # let validation error bubble up
        if self.cls_factory is not None:
            return self.cls_factory.model_validate(sel_data)
        else:
            raise ValueError(
                "Class factory should be set to a BaseModel subclass before calling"
            )


def ResourceConfig(
    config_file: str,
    path_selector: str | None = None,
) -> _ResourceConfig:
    """
    Wrapper for a _ResourceConfig.

    Attributes:
        config_file (str): JSON file where the configuration is stored
        path_selector (str | None): Path selector to retrieve a specific value from the JSON map
        cache (bool): Cache the resource's value to avoid re-computation.

    Returns:
        _ResourceConfig: A configured resource representation
    """

    return _ResourceConfig(config_file=config_file, path_selector=path_selector)


class ResourceDefinition(BaseModel):
    """Definition for a resource injection requested by a step signature.

    Attributes:
        name (str): Parameter name in the step function.
        resource (_Resource): Factory wrapper used by the manager to produce the dependency.
        type_annotation (type | None): The type annotation from Annotated[T, Resource(...)].
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    resource: _Resource
    type_annotation: Any = None


def Resource(
    factory: Callable[..., T],
    cache: bool = True,
) -> _Resource:
    """Declare a resource to inject into step functions.

    Args:
        factory (Callable[..., T] | None): Function returning the resource instance. May be async.
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
