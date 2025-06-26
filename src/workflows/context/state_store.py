import asyncio
import warnings
from pydantic import BaseModel
from typing import Any, Generic, Optional, TypeVar

from workflows.events import Event
from .serializers import BaseSerializer

MAX_DEPTH = 1000
MODEL_T = TypeVar("MODEL_T", bound=BaseModel)


# Only warn once about unserializable keys
class UnserializableKeyWarning(Warning):
    pass


warnings.simplefilter("once", UnserializableKeyWarning)


class DictState(Event):
    """
    A dynamic state class that behaves like a dictionary.

    This is used as the default state type when no specific state class is provided.
    It allows storing arbitrary key-value pairs while still being a Pydantic model.
    """

    pass


class InMemoryStateStore(Generic[MODEL_T]):
    """
    State manager for a workflow that provides type-safe state management.

    By using Context[StateType] as the parameter type annotation, the state manager
    is automatically initialized with the correct type, providing full type safety
    and IDE autocompletion.

    When no state type is specified (just Context), it defaults to DictState which
    behaves like a regular dictionary.

    Example with typed state:
    ```python
    from pydantic import BaseModel
    from workflows import Workflow, Context, step
    from workflows.events import StartEvent, StopEvent

    class MyState(BaseModel):
        name: str = "Unknown"
        age: int = 0

    class MyWorkflow(Workflow):
        @step
        async def step_1(self, ctx: Context[MyState], ev: StartEvent) -> StopEvent:
            # ctx._state.get() is now properly typed as MyState
            state = await ctx._state.get()
            state.name = "John"  # Type-safe: IDE knows this is a string
            state.age = 30       # Type-safe: IDE knows this is an int
            await ctx._state.set(state)
            return StopEvent()
    ```

    Example with untyped dict-like state:
    ```python
    class MyWorkflow(Workflow):
        @step
        async def step_1(self, ctx: Context, ev: StartEvent) -> StopEvent:
            # ctx._state behaves like a dict
            state = await ctx._state.get()
            state.name = "John"     # Works like a dict
            state.age = 30          # Dynamic assignment
            await ctx._state.set(state)
            return StopEvent()
    ```

    The state manager provides:
    - Type-safe access to state properties with full IDE support (when typed)
    - Dict-like behavior for dynamic state management (when untyped)
    - Automatic state initialization based on the generic type parameter
    - Thread-safe state access with async locking
    - Deep path-based state access and modification
    """

    # These keys are set by pre-built workflows and
    # are known to be unserializable in some cases.
    known_unserializable_keys = ("memory",)

    def __init__(self, initial_state: MODEL_T):
        self._state = initial_state
        self._lock = asyncio.Lock()

    async def get_state(self) -> MODEL_T:
        """Get a copy of the current state."""
        return self._state.model_copy()

    async def set_state(self, state: MODEL_T) -> None:
        """Set the current state."""
        if not isinstance(state, type(self._state)):
            raise ValueError(f"State must be of type {type(self._state)}")

        async with self._lock:
            self._state = state

    def to_dict(self, serializer: "BaseSerializer") -> dict[str, Any]:
        """
        Serialize the state manager's state.

        For DictState, uses the BaseSerializer for individual items since they can be arbitrary types.
        For other Pydantic models, leverages Pydantic's serialization but uses BaseSerializer for complex types.
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
        """
        Deserialize and restore a state manager.
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

    async def get(self, path: str, default: Optional[Any] = Ellipsis) -> Any:
        """
        Return a value from *path*, where path is a dot-separated string.
        Example: await sm.get("user.profile.name")
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
        """Set *value* at the location designated by *path* (dot-separated)."""
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
