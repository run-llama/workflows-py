# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Generic,
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

    def get_resource_configs(
        self, resource_configs_cache: dict[str, dict[str, BaseModel]]
    ) -> dict[str, BaseModel]:
        params = inspect.signature(self._factory).parameters
        resource_configs: dict[str, BaseModel] = {}
        if len(params) > 0:
            for param in params.values():
                if get_origin(param.annotation) is Annotated:
                    args = get_args(param.annotation)
                    if len(args) == 2 and isinstance(args[1], _ResourceConfig):
                        resource_config = args[1]
                        if (
                            resource_config.cache
                            and (
                                cached := resource_configs_cache.get(self.name, {}).get(
                                    resource_config.name
                                )
                            )
                            is not None
                        ):
                            value = cached
                        else:
                            resource_config.cls_factory = args[0]
                            value = resource_config.call()
                            if resource_config.cache:
                                if self.name not in resource_configs_cache:
                                    resource_configs_cache[self.name] = {
                                        resource_config.name: value
                                    }
                                else:
                                    resource_configs_cache[self.name][
                                        resource_config.name
                                    ] = value
                        resource_configs.update({param.name: value})
        return resource_configs

    async def call(self, resources_config: dict[str, BaseModel] | None = None) -> T:
        """Invoke the underlying factory, awaiting if necessary."""
        args = resources_config or {}
        if self._is_async:
            result = await cast(Callable[..., Awaitable[T]], self._factory)(**args)
        else:
            result = cast(Callable[..., T], self._factory)(**args)
        return result


class _ResourceConfig(Generic[B]):
    """
    Internal wrapper for a pydantic-based resource whose configuration can be read from a JSON file.
    """

    def __init__(
        self,
        config_file: str,
        path_selector: str | None,
        cache: bool,
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
        self.cache = cache

    @property
    def name(self) -> str:
        if self.path_selector is not None:
            return self.config_file + "." + self.path_selector
        return self.config_file

    def _get_data(self, data: dict[str, Any]) -> dict[str, Any] | None:
        if self.path_selector is not None:
            keys = self.path_selector.split(".")
            val: dict[str, Any] | None = data
            for key in keys:
                if isinstance(val, dict):
                    val = cast(dict[str, Any] | None, val.get(key))
                    if val is None:
                        return None
                else:
                    return None
            return val
        return data

    # make async for compatibility with _Resource
    def call(self) -> B:
        with open(self.config_file, "r") as f:
            data = json.load(f)
        if (sel_data := self._get_data(data)) is not None:
            # let validation error bubble up
            if self.cls_factory is not None:
                return self.cls_factory.model_validate(sel_data)
            raise ValueError(
                "Class factory should be set to a BaseModel subclass before calling"
            )
        raise ValueError(
            f"Invalid path selector for {self.config_file}: {self.path_selector}"
        )


def ResourceConfig(
    config_file: str,
    path_selector: str | None = None,
    cache: bool = True,
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

    return _ResourceConfig(
        config_file=config_file, path_selector=path_selector, cache=cache
    )


class ResourceDefinition(BaseModel):
    """Definition for a resource injection requested by a step signature.

    Attributes:
        name (str): Parameter name in the step function.
        resource (_Resource): Factory wrapper used by the manager to produce the dependency.
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
        self.resources_config: dict[str, dict[str, BaseModel]] = {}

    async def set(self, name: str, val: Any) -> None:
        """Register a resource instance under a name."""
        self.resources.update({name: val})

    async def get(self, resource: _Resource) -> Any:
        """Return a resource instance, honoring cache settings."""
        resources_config = resource.get_resource_configs(
            resource_configs_cache=self.resources_config
        )
        config = resources_config if len(resources_config) > 0 else None
        if not resource.cache:
            val = await resource.call(config)
        elif resource.cache and not self.resources.get(resource.name, None):
            val = await resource.call(config)
            await self.set(resource.name, val)
        else:
            val = self.resources.get(resource.name)
        return val

    def get_resource_config(
        self, resource_name: str, resource_config: _ResourceConfig
    ) -> BaseModel | None:
        """Return a specific cached resource config"""
        return self.resources_config.get(resource_name, {}).get(resource_config.name)

    def get_all_resource_configs(
        self, resource_name: str
    ) -> dict[str, BaseModel] | None:
        """Get all the cached configurations for a given resource"""
        return self.resources_config.get(resource_name)

    def get_all(self) -> dict[str, Any]:
        """Return all materialized resources."""
        return self.resources
