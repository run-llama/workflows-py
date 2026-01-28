# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import asyncio
import functools
import warnings
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generic, Type

from pydantic import BaseModel, ValidationError
from typing_extensions import TypeVar

from workflows.decorators import StepConfig
from workflows.events import DictLikeModel

from .serializers import BaseSerializer

if TYPE_CHECKING:
    from workflows.workflow import Workflow

MAX_DEPTH = 1000


# Only warn once about unserializable keys
class UnserializableKeyWarning(Warning):
    pass


warnings.simplefilter("once", UnserializableKeyWarning)


class DictState(DictLikeModel):
    """
    Dynamic, dict-like Pydantic model for workflow state.

    Used as the default state model when no typed state is provided. Behaves
    like a mapping while retaining Pydantic validation and serialization.

    Examples:
        ```python
        from workflows.context.state_store import DictState

        state = DictState()
        state["foo"] = 1
        state.bar = 2  # attribute-style access works for nested structures
        ```

    See Also:
        - [InMemoryStateStore][workflows.context.state_store.InMemoryStateStore]
    """

    def __init__(self, **params: Any):
        super().__init__(**params)


# Default state type is DictState for the generic type
MODEL_T = TypeVar("MODEL_T", bound=BaseModel, default=DictState)  # type: ignore


class InMemoryStateStore(Generic[MODEL_T]):
    """
    Async, in-memory, type-safe state manager for workflows.

    This store holds a single Pydantic model instance representing global
    workflow state. When the generic parameter is omitted, it defaults to
    [DictState][workflows.context.state_store.DictState] for flexible,
    dictionary-like usage.

    Thread-safety is ensured with an internal `asyncio.Lock`. Consumers can
    either perform atomic reads/writes via `get_state` and `set_state`, or make
    in-place, transactional edits via the `edit_state` context manager.

    Examples:
        Typed state model:

        ```python
        from pydantic import BaseModel
        from workflows.context.state_store import InMemoryStateStore

        class MyState(BaseModel):
            count: int = 0

        store = InMemoryStateStore(MyState())
        async with store.edit_state() as state:
            state.count += 1
        ```

        Dynamic state with `DictState`:

        ```python
        from workflows.context.state_store import InMemoryStateStore, DictState

        store = InMemoryStateStore(DictState())
        await store.set("user.profile.name", "Ada")
        name = await store.get("user.profile.name")
        ```

    See Also:
        - [Context.store][workflows.context.context.Context.store]
    """

    # These keys are set by pre-built workflows and
    # are known to be unserializable in some cases.
    known_unserializable_keys = ("memory",)

    state_type: Type[MODEL_T]

    def __init__(self, initial_state: MODEL_T):
        self._state = initial_state
        self.state_type = type(initial_state)

    @functools.cached_property
    def _lock(self) -> asyncio.Lock:
        """Lazy lock initialization for Python 3.14+ compatibility.

        asyncio.Lock() requires a running event loop in Python 3.14+.
        Using cached_property defers creation to first use in async context.
        """
        return asyncio.Lock()

    async def get_state(self) -> MODEL_T:
        """Return a shallow copy of the current state model.

        Returns:
            MODEL_T: A `.model_copy()` of the internal Pydantic model.
        """
        return self._state.model_copy()

    async def set_state(self, state: MODEL_T) -> None:
        """Replace or merge into the current state model.

        If the provided state is the exact type of the current state, it replaces
        the state entirely. If the provided state is a parent type (i.e., the
        current state type is a subclass of the provided state type), the fields
        from the parent are merged onto the current state, preserving any child
        fields that aren't present in the parent.

        This enables workflow inheritance where a base workflow step can call
        set_state with a base state type without obliterating child state fields.

        Args:
            state (MODEL_T): New state, either the same type or a parent type.

        Raises:
            ValueError: If the types are not compatible (neither same nor parent).
        """
        current_type = type(self._state)
        new_type = type(state)

        if isinstance(state, current_type):
            # Exact match or subclass - direct replacement
            async with self._lock:
                self._state = state
        elif issubclass(current_type, new_type):
            # Parent type provided - merge fields onto current state
            # This preserves child-specific fields while updating parent fields
            async with self._lock:
                # Get the fields from the parent type and update them on the current state
                parent_data = state.model_dump()
                self._state = current_type.model_validate(
                    {**self._state.model_dump(), **parent_data}
                )
        else:
            raise ValueError(
                f"State must be of type {current_type.__name__} or a parent type, "
                f"got {new_type.__name__}"
            )

    def to_dict(self, serializer: "BaseSerializer") -> dict[str, Any]:
        """Serialize the state and model metadata for persistence.

        For `DictState`, each individual item is serialized using the provided
        serializer since values can be arbitrary Python objects. For other
        Pydantic models, defers to the serializer (e.g. JSON) which can leverage
        model-aware encoding.

        Args:
            serializer (BaseSerializer): Strategy used to encode values.

        Returns:
            dict[str, Any]: A payload suitable for
            [from_dict][workflows.context.state_store.InMemoryStateStore.from_dict].
        """
        # Special handling for DictState - serialize each item in _data
        if isinstance(self._state, DictState):
            serialized_data = {}
            for key, value in self._state.items():
                try:
                    serialized_data[key] = serializer.serialize(value)
                except Exception as e:
                    if key in self.known_unserializable_keys:
                        warnings.warn(
                            f"Skipping serialization of known unserializable key: {key} -- "
                            "This is expected but will require this item to be set manually after deserialization.",
                            category=UnserializableKeyWarning,
                        )
                        continue
                    raise ValueError(
                        f"Failed to serialize state value for key {key}: {e}"
                    )

            return {
                "state_data": {"_data": serialized_data},
                "state_type": type(self._state).__name__,
                "state_module": type(self._state).__module__,
            }
        else:
            # For regular Pydantic models, rely on pydantic's serialization
            serialized_state = serializer.serialize(self._state)

            return {
                "state_data": serialized_state,
                "state_type": type(self._state).__name__,
                "state_module": type(self._state).__module__,
            }

    @classmethod
    def from_dict(
        cls, serialized_state: dict[str, Any], serializer: "BaseSerializer"
    ) -> "InMemoryStateStore[MODEL_T]":
        """Restore a state store from a serialized payload.

        Args:
            serialized_state (dict[str, Any]): The payload produced by
                [to_dict][workflows.context.state_store.InMemoryStateStore.to_dict].
            serializer (BaseSerializer): Strategy to decode stored values.

        Returns:
            InMemoryStateStore[MODEL_T]: A store with the reconstructed model.
        """
        if not serialized_state:
            # Return a default DictState manager
            return cls(DictState())  # type: ignore

        state_data = serialized_state.get("state_data", {})
        state_type = serialized_state.get("state_type", "DictState")

        # Deserialize the state data
        if state_type == "DictState":
            # Special handling for DictState - deserialize each item in _data
            _data_serialized = state_data.get("_data", {})
            deserialized_data = {}
            for key, value in _data_serialized.items():
                try:
                    deserialized_data[key] = serializer.deserialize(value)
                except Exception as e:
                    raise ValueError(
                        f"Failed to deserialize state value for key {key}: {e}"
                    )

            state_instance = DictState(_data=deserialized_data)
        else:
            state_instance = serializer.deserialize(state_data)

        return cls(state_instance)  # type: ignore

    @asynccontextmanager
    async def edit_state(self) -> AsyncGenerator[MODEL_T, None]:
        """Edit state transactionally under a lock.

        Yields the mutable model and writes it back on exit. This pattern avoids
        read-modify-write races and keeps updates atomic.

        Yields:
            MODEL_T: The current state model for in-place mutation.
        """
        async with self._lock:
            state = self._state

            yield state

            self._state = state

    async def get(self, path: str, default: Any = Ellipsis) -> Any:
        """Get a nested value using dot-separated paths.

        Supports dict keys, list indices, and attribute access transparently at
        each segment.

        Args:
            path (str): Dot-separated path, e.g. "user.profile.name".
            default (Any): If provided, return this when the path does not
                exist; otherwise, raise `ValueError`.

        Returns:
            Any: The resolved value.

        Raises:
            ValueError: If the path is invalid and no default is provided or if
                the path depth exceeds limits.
        """
        segments = path.split(".") if path else []
        if len(segments) > MAX_DEPTH:
            raise ValueError(f"Path length exceeds {MAX_DEPTH} segments")

        async with self._lock:
            try:
                value: Any = self._state
                for segment in segments:
                    value = self._traverse_step(value, segment)
            except Exception:
                if default is not Ellipsis:
                    return default

                msg = f"Path '{path}' not found in state"
                raise ValueError(msg)

        return value

    async def set(self, path: str, value: Any) -> None:
        """Set a nested value using dot-separated paths.

        Intermediate containers are created as needed. Dicts, lists, tuples, and
        Pydantic models are supported where appropriate.

        Args:
            path (str): Dot-separated path to write.
            value (Any): Value to assign.

        Raises:
            ValueError: If the path is empty or exceeds the maximum depth.
        """
        if not path:
            raise ValueError("Path cannot be empty")

        segments = path.split(".")
        if len(segments) > MAX_DEPTH:
            raise ValueError(f"Path length exceeds {MAX_DEPTH} segments")

        async with self._lock:
            current = self._state

            # Navigate/create intermediate segments
            for segment in segments[:-1]:
                try:
                    current = self._traverse_step(current, segment)
                except (KeyError, AttributeError, IndexError, TypeError):
                    # Create intermediate object and assign it
                    intermediate: Any = {}
                    self._assign_step(current, segment, intermediate)
                    current = intermediate

            # Assign the final value
            self._assign_step(current, segments[-1], value)

    async def clear(self) -> None:
        """Reset the state to its type defaults.

        Raises:
            ValueError: If the model type cannot be instantiated from defaults
                (i.e., fields missing default values).
        """
        try:
            await self.set_state(self._state.__class__())
        except ValidationError:
            raise ValueError("State must have defaults for all fields")

    def _traverse_step(self, obj: Any, segment: str) -> Any:
        """Follow one segment into *obj* (dict key, list index, or attribute)."""
        if isinstance(obj, dict):
            return obj[segment]

        # attempt list/tuple index
        try:
            idx = int(segment)
            return obj[idx]
        except (ValueError, TypeError, IndexError):
            pass

        # fallback to attribute access (Pydantic models, normal objects)
        return getattr(obj, segment)

    def _assign_step(self, obj: Any, segment: str, value: Any) -> None:
        """Assign *value* to *segment* of *obj* (dict key, list index, or attribute)."""
        if isinstance(obj, dict):
            obj[segment] = value
            return

        # attempt list/tuple index assignment
        try:
            idx = int(segment)
            obj[idx] = value
            return
        except (ValueError, TypeError, IndexError):
            pass

        # fallback to attribute assignment
        setattr(obj, segment, value)


def infer_state_type(workflow: "Workflow") -> type[BaseModel]:
    """Infer the state type from workflow step configs.

    Looks at Context[T] annotations in step functions to determine
    the expected state type. Returns DictState if no typed state is found.

    Args:
        workflow: The workflow to inspect for state type annotations.

    Returns:
        The inferred state type, or DictState if none found.

    Raises:
        ValueError: If multiple different state types are found.
    """
    state_types: set[type[BaseModel]] = set()
    for _, step_func in workflow._get_steps().items():
        step_config: StepConfig = step_func._step_config
        if (
            step_config.context_state_type is not None
            and step_config.context_state_type != DictState
            and issubclass(step_config.context_state_type, BaseModel)
        ):
            state_types.add(step_config.context_state_type)

    state_type: Type[BaseModel]
    if state_types:
        state_type = _find_most_derived_state_type(state_types)
    else:
        state_type = DictState

    return state_type


def _find_most_derived_state_type(state_types: set[Type[BaseModel]]) -> Type[BaseModel]:
    """Find the most derived (most specific) state type from a set of types.

    All types must be in a single inheritance chain, i.e., one type must be
    a subclass of all other types (the most derived type).

    Args:
        state_types: Set of state types to analyze.

    Returns:
        The most derived type in the inheritance hierarchy.

    Raises:
        ValueError: If types are not in a compatible inheritance hierarchy.
    """
    type_list = list(state_types)

    if len(type_list) == 1:
        return type_list[0]

    # Find the most derived type - it should be a subclass of all others
    most_derived: Type[BaseModel] | None = None

    for candidate in type_list:
        is_most_derived = True
        for other in type_list:
            if other is candidate:
                continue
            # candidate must be a subclass of other (or equal to it)
            if not issubclass(candidate, other):
                is_most_derived = False
                break
        if is_most_derived:
            most_derived = candidate
            break

    if most_derived is None:
        # No single type is a subclass of all others - incompatible hierarchy
        raise ValueError(
            "Multiple state types are not in a compatible inheritance hierarchy. "
            "All state types must share a common inheritance chain. Found: "
            + ", ".join([st.__name__ for st in state_types])
        )

    return most_derived
