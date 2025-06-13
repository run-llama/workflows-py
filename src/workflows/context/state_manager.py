import asyncio
from pydantic import BaseModel
from typing import Any, Dict, Generic, TypeVar, Union

MAX_DEPTH = 1000
T = TypeVar("T", bound=BaseModel)


class InMemoryStateManager(Generic[T]):
    """
    State manager for a workflow.

    By annotating a the Context object with a state class, you can use the state manager to get and set state.

    Example:
    ```python
    class MyState(BaseModel):
        name: str
        age: int
    
    class MyWorkflow(Workflow):
        @step
        async def step_1(self, ctx: Context[MyState], ev: StartEvent) -> StopEvent:
            state = await ctx.state.get()
            state.name = "John"
            state.age = 30
            await ctx.state.set(state)

            return StopEvent()
            
    """
    def __init__(self, initial_state: T):
        self.state = initial_state
        self._lock = asyncio.Lock()
    
    async def get(self) -> T:
        """Get a copy of the current state."""
        return self.state.model_copy()
    
    async def set(self, state: T) -> None:
        """Set the current state."""
        async with self._lock:
            self.state = state
    
    async def update(self, updated_state: Union[T, Dict[str, Any]]) -> None:
        """Update the current state (only modifies what has changed)."""
        async with self._lock:
            if isinstance(updated_state, dict):
                self.state = self.state.model_copy(update=updated_state)
            else:
                self.state = self.state.model_copy(update=updated_state.model_dump())
    
    async def get_path(self, path: str) -> Any:
        """
        Return a value from *path*, where path is a dot-separated string.

        Example: await sm.get_path("user.profile.name")
        """
        segments = path.split(".") if path else []
        if len(segments) > MAX_DEPTH:
            raise ValueError(f"Path length exceeds {MAX_DEPTH} segments")
        
        async with self._lock:
            value: Any = self.state
            for segment in segments:
                value = self._traverse_step(value, segment)
        
        return value

    async def set_path(self, path: str, value: Any) -> None:
        """Set *value* at the location designated by *path* (dot-separated)."""
        if not path:
            raise ValueError("Path cannot be empty")
        
        segments = path.split(".")
        if len(segments) > MAX_DEPTH:
            raise ValueError(f"Path length exceeds {MAX_DEPTH} segments")
        
        async with self._lock:
            target: Any = self.state.model_dump()
            for segment in segments[:-1]:
                target = self._traverse_step(target, segment)
            
            # assign the last segment
            self._assign_step(target, segments[-1], value)

            # update the state
            self.state = self.state.model_copy(update=target)

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
        