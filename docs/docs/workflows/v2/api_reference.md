<a id="workflows.context.state_store.DictState"></a>

## DictState

```python
class DictState(DictLikeModel)
```

Dynamic, dict-like Pydantic model for workflow state.

Used as the default state model when no typed state is provided. Behaves
like a mapping while retaining Pydantic validation and serialization.

**Examples**:

```python
from workflows.context.state_store import DictState

state = DictState()
state["foo"] = 1
state.bar = 2  # attribute-style access works for nested structures
```


**See Also**:

  - [`InMemoryStateStore`](#workflows.context.state_store.InMemoryStateStore)

<a id="workflows.context.state_store.InMemoryStateStore"></a>

## InMemoryStateStore

```python
class InMemoryStateStore(Generic[MODEL_T])
```

Async, in-memory, type-safe state manager for workflows.

This store holds a single Pydantic model instance representing global
workflow state. When the generic parameter is omitted, it defaults to
[`DictState`](#workflows.context.state_store.DictState) for flexible, dictionary-like usage.

Thread-safety is ensured with an internal `asyncio.Lock`. Consumers can
either perform atomic reads/writes via [`get_state`](#workflows.context.state_store.InMemoryStateStore.get_state) and [`set_state`](#workflows.context.state_store.InMemoryStateStore.set_state), or make
in-place, transactional edits via the [`edit_state`](#workflows.context.state_store.InMemoryStateStore.edit_state) context manager.

**Examples**:

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


**See Also**:

  - [Context.store](#workflows.context.context.Context.store)

<a id="workflows.context.state_store.InMemoryStateStore.get_state"></a>

#### get\_state

```python
async def get_state() -> MODEL_T
```

Return a shallow copy of the current state model.

**Returns**:

- `MODEL_T` - A `.model_copy()` of the internal Pydantic model.

<a id="workflows.context.state_store.InMemoryStateStore.set_state"></a>

#### set\_state

```python
async def set_state(state: MODEL_T) -> None
```

Replace the current state model.

**Arguments**:

- `state` _MODEL_T_ - New state of the same type as the existing model.


**Raises**:

- `ValueError` - If the type differs from the existing state type.

<a id="workflows.context.state_store.InMemoryStateStore.to_dict"></a>

#### to\_dict

```python
def to_dict(serializer: "BaseSerializer") -> dict[str, Any]
```

Serialize the state and model metadata for persistence.

For `DictState`, each individual item is serialized using the provided
serializer since values can be arbitrary Python objects. For other
Pydantic models, defers to the serializer (e.g. JSON) which can leverage
model-aware encoding.

**Arguments**:

- `serializer` _BaseSerializer_ - Strategy used to encode values.


**Returns**:

- `dict` - A payload suitable for [`from_dict`](#workflows.context.state_store.InMemoryStateStore.from_dict).

<a id="workflows.context.state_store.InMemoryStateStore.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, serialized_state: dict[str, Any],
              serializer: "BaseSerializer") -> "InMemoryStateStore[MODEL_T]"
```

Restore a state store from a serialized payload.

**Arguments**:

- `serialized_state` _dict[str, Any]_ - The payload produced by [`to_dict`](#workflows.context.state_store.InMemoryStateStore.to_dict).
- `serializer` _BaseSerializer_ - Strategy to decode stored values.


**Returns**:

- `InMemoryStateStore[MODEL_T]` - A store with the reconstructed model.

<a id="workflows.context.state_store.InMemoryStateStore.edit_state"></a>

#### edit\_state

```python
@asynccontextmanager
async def edit_state() -> AsyncGenerator[MODEL_T, None]
```

Edit state transactionally under a lock.

Yields the mutable model and writes it back on exit. This pattern avoids
read-modify-write races and keeps updates atomic.

**Yields**:

- `MODEL_T` - The current state model for in-place mutation.

<a id="workflows.context.state_store.InMemoryStateStore.get"></a>

#### get

```python
async def get(path: str, default: Optional[Any] = Ellipsis) -> Any
```

Get a nested value using dot-separated paths.

Supports dict keys, list indices, and attribute access transparently at
each segment.

**Arguments**:

- `path` _str_ - Dot-separated path, e.g. "user.profile.name".
- `default` _Any_ - If provided, return this when the path does not
  exist; otherwise, raise `ValueError`.


**Returns**:

- `Any` - The resolved value.


**Raises**:

- `ValueError` - If the path is invalid and no default is provided or if
  the path depth exceeds limits.

<a id="workflows.context.state_store.InMemoryStateStore.set"></a>

#### set

```python
async def set(path: str, value: Any) -> None
```

Set a nested value using dot-separated paths.

Intermediate containers are created as needed. Dicts, lists, tuples, and
Pydantic models are supported where appropriate.

**Arguments**:

- `path` _str_ - Dot-separated path to write.
- `value` _Any_ - Value to assign.


**Raises**:

- `ValueError` - If the path is empty or exceeds the maximum depth.

<a id="workflows.context.state_store.InMemoryStateStore.clear"></a>

#### clear

```python
async def clear() -> None
```

Reset the state to its type defaults.

**Raises**:

- `ValueError` - If the model type cannot be instantiated from defaults
  (i.e., fields missing default values).

<a id="workflows.context.serializers.BaseSerializer"></a>

## BaseSerializer

```python
class BaseSerializer(ABC)
```

Interface for value serialization used by the workflow context and state store.

Implementations must encode arbitrary Python values into a string and be able
to reconstruct the original values from that string.

**See Also**:

  - [`JsonSerializer`](#workflows.context.serializers.JsonSerializer)
  - [`PickleSerializer`](#workflows.context.serializers.PickleSerializer)

<a id="workflows.context.serializers.JsonSerializer"></a>

## JsonSerializer

```python
class JsonSerializer(BaseSerializer)
```

JSON-first serializer that understands Pydantic models and LlamaIndex components.

Behavior:
- Pydantic models are encoded as JSON with their qualified class name so they
can be faithfully reconstructed.
- LlamaIndex components (objects exposing `class_name` and `to_dict`) are
serialized to their dict form alongside the qualified class name.
- Dicts and lists are handled recursively.

Fallback for unsupported objects is to attempt JSON encoding directly; if it
fails, a `ValueError` is raised.

**Examples**:

```python
s = JsonSerializer()
payload = s.serialize({"x": 1, "y": [2, 3]})
data = s.deserialize(payload)
assert data == {"x": 1, "y": [2, 3]}
```


**See Also**:

  - [`BaseSerializer`](#workflows.context.serializers.BaseSerializer)
  - [`PickleSerializer`](#workflows.context.serializers.PickleSerializer)

<a id="workflows.context.serializers.JsonSerializer.serialize"></a>

#### serialize

```python
def serialize(value: Any) -> str
```

Serialize an arbitrary value to a JSON string.

**Arguments**:

- `value` _Any_ - The value to encode.


**Returns**:

- `str` - JSON string.


**Raises**:

- `ValueError` - If the value cannot be encoded to JSON.

<a id="workflows.context.serializers.JsonSerializer.deserialize"></a>

#### deserialize

```python
def deserialize(value: str) -> Any
```

Deserialize a JSON string into Python objects.

**Arguments**:

- `value` _str_ - JSON string.


**Returns**:

- `Any` - The reconstructed value.

<a id="workflows.context.serializers.PickleSerializer"></a>

## PickleSerializer

```python
class PickleSerializer(JsonSerializer)
```

Hybrid serializer: JSON when possible, Pickle as a safe fallback.

This serializer attempts JSON first for readability and portability, and
transparently falls back to Pickle for objects that cannot be represented in
JSON. Deserialization prioritizes Pickle and falls back to JSON.

**Warnings**:

  Pickle can execute arbitrary code during deserialization. Only
  deserialize trusted payloads.

- `Note` - Used to be called `JsonPickleSerializer` but it was renamed to `PickleSerializer`.


**Examples**:

```python
s = PickleSerializer()
class Foo:
    def __init__(self, x):
        self.x = x
payload = s.serialize(Foo(1))  # will likely use Pickle
obj = s.deserialize(payload)
assert isinstance(obj, Foo)
```

<a id="workflows.context.serializers.PickleSerializer.serialize"></a>

#### serialize

```python
def serialize(value: Any) -> str
```

Serialize with JSON preference and Pickle fallback.

**Arguments**:

- `value` _Any_ - The value to encode.


**Returns**:

- `str` - Encoded string (JSON or base64-encoded Pickle bytes).

<a id="workflows.context.serializers.PickleSerializer.deserialize"></a>

#### deserialize

```python
def deserialize(value: str) -> Any
```

Deserialize with Pickle preference and JSON fallback.

**Arguments**:

- `value` _str_ - Encoded string.


**Returns**:

- `Any` - The reconstructed value.


**Notes**:

  Use only with trusted payloads due to Pickle security implications.

<a id="workflows.context.context.Context"></a>

## Context

```python
class Context(Generic[MODEL_T])
```

Global, per-run context and event broker for a `Workflow`.

The `Context` coordinates event delivery between steps, tracks in-flight work,
exposes a global state store, and provides utilities for streaming and
synchronization. It is created by a `Workflow` at run time and can be
persisted and restored.

**Arguments**:

- `workflow` _Workflow_ - The owning workflow instance. Used to infer
  step configuration and instrumentation.


**Attributes**:

- `is_running` _bool_ - Whether the workflow is currently running.
- `store` _InMemoryStateStore[MODEL_T]_ - Type-safe, async state store shared
  across steps. See also
  [InMemoryStateStore](#workflows.context.state_store.InMemoryStateStore).


**Examples**:

  Basic usage inside a step:

```python
from workflows import step
from workflows.events import StartEvent, StopEvent

@step
async def start(self, ctx: Context, ev: StartEvent) -> StopEvent:
    await ctx.store.set("query", ev.topic)
    ctx.write_event_to_stream(ev)  # surface progress to UI
    return StopEvent(result="ok")
```

  Persisting the state of a workflow across runs:

```python
from workflows import Context

# Create a context and run the workflow with the same context
ctx = Context(my_workflow)
result_1 = await my_workflow.run(..., ctx=ctx)
result_2 = await my_workflow.run(..., ctx=ctx)

# Serialize the context and restore it
ctx_dict = ctx.to_dict()
restored_ctx = Context.from_dict(my_workflow, ctx_dict)
result_3 = await my_workflow.run(..., ctx=restored_ctx)
```



**See Also**:

  - [Workflow](#workflows.workflow.Workflow)
  - [Event](#workflows.events.Event)
  - [InMemoryStateStore](#workflows.context.state_store.InMemoryStateStore)

<a id="workflows.context.context.Context.store"></a>

#### store

```python
@property
def store() -> InMemoryStateStore[MODEL_T]
```

Typed, process-local state store shared across steps.

If no state was initialized yet, a default
[DictState](#workflows.context.state_store.DictState) store is created.

**Returns**:

- `InMemoryStateStore[MODEL_T]` - The state store instance.

<a id="workflows.context.context.Context.to_dict"></a>

#### to\_dict

```python
def to_dict(serializer: BaseSerializer | None = None) -> dict[str, Any]
```

Serialize the context to a JSON-serializable dict.

Persists the global state store, event queues, buffers, accepted events,
broker log, and running flag. This payload can be fed to
[from_dict](#workflows.context.context.Context.from_dict) to resume a run
or carry state across runs.

**Arguments**:

- `serializer` _BaseSerializer | None_ - Value serializer used for state
  and event payloads. Defaults to
  [JsonSerializer](#workflows.context.serializers.JsonSerializer).


**Returns**:

- `dict` - A dict suitable for JSON encoding and later restoration via `from_dict`.


**See Also**:

  - [InMemoryStateStore.to_dict](#workflows.context.state_store.InMemoryStateStore.to_dict)


**Examples**:

```python
ctx_dict = ctx.to_dict()
my_db.set("key", json.dumps(ctx_dict))

ctx_dict = my_db.get("key")
restored_ctx = Context.from_dict(my_workflow, json.loads(ctx_dict))
result = await my_workflow.run(..., ctx=restored_ctx)
```

<a id="workflows.context.context.Context.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls,
              workflow: "Workflow",
              data: dict[str, Any],
              serializer: BaseSerializer | None = None) -> "Context[MODEL_T]"
```

Reconstruct a `Context` from a serialized payload.

**Arguments**:

- `workflow` _Workflow_ - The workflow instance that will own this
  context.
- `data` _dict[str, Any]_ - Payload produced by
  [to_dict](#workflows.context.context.Context.to_dict).
- `serializer` _BaseSerializer | None_ - Serializer used to decode state
  and events. Defaults to JSON.


**Returns**:

- `Context[MODEL_T]` - A context instance initialized with the persisted
  state and queues.


**Raises**:

- `ContextSerdeError` - If the payload is missing required fields or is
  in an incompatible format.


**Examples**:

```python
ctx_dict = ctx.to_dict()
my_db.set("key", json.dumps(ctx_dict))

ctx_dict = my_db.get("key")
restored_ctx = Context.from_dict(my_workflow, json.loads(ctx_dict))
result = await my_workflow.run(..., ctx=restored_ctx)
```

<a id="workflows.context.context.Context.mark_in_progress"></a>

#### mark\_in\_progress

```python
async def mark_in_progress(name: str, ev: Event) -> None
```

Add input event to in_progress dict.

**Arguments**:

- `name` _str_ - The name of the step that is in progress.
- `ev` _Event_ - The input event that kicked off this step.

<a id="workflows.context.context.Context.remove_from_in_progress"></a>

#### remove\_from\_in\_progress

```python
async def remove_from_in_progress(name: str, ev: Event) -> None
```

Remove input event from active steps.

**Arguments**:

- `name` _str_ - The name of the step that has been completed.
- `ev` _Event_ - The associated input event that kicked of this completed step.

<a id="workflows.context.context.Context.running_steps"></a>

#### running\_steps

```python
async def running_steps() -> list[str]
```

Return the list of currently running step names.

**Returns**:

- `list[str]` - Names of steps that have at least one active worker.

<a id="workflows.context.context.Context.lock"></a>

#### lock

```python
@property
def lock() -> asyncio.Lock
```

Returns a mutex to lock the Context.

<a id="workflows.context.context.Context.collect_events"></a>

#### collect\_events

```python
def collect_events(ev: Event,
                   expected: list[Type[Event]],
                   buffer_id: str | None = None) -> list[Event] | None
```

Buffer events until all expected types are available, then return them.

This utility is helpful when a step can receive multiple event types
and needs to proceed only when it has a full set. The returned list is
ordered according to `expected`.

**Arguments**:

- `ev` _Event_ - The incoming event to add to the buffer.
- `expected` _list[Type[Event]]_ - Event types to collect, in order.
- `buffer_id` _str | None_ - Optional stable key to isolate buffers across
  steps or workers. Defaults to an internal key derived from the
  task name or expected types.


**Returns**:

  list | None: The list of events in the requested order when complete,
  otherwise `None`.


**Examples**:

```python
@step
async def synthesize(
    self, ctx: Context, ev: QueryEvent | RetrieveEvent
) -> StopEvent | None:
    events = ctx.collect_events(ev, [QueryEvent, RetrieveEvent])
    if events is None:
        return None
    query_ev, retrieve_ev = events
    # ... proceed with both inputs present ...
```


**See Also**:

  - [Event](#workflows.events.Event)

<a id="workflows.context.context.Context.send_event"></a>

#### send\_event

```python
def send_event(message: Event, step: str | None = None) -> None
```

Dispatch an event to one or all workflow steps.

If `step` is omitted, the event is broadcast to all step queues and
non-matching steps will ignore it. When `step` is provided, the target
step must accept the event type or a
[WorkflowRuntimeError](#workflows.errors.WorkflowRuntimeError) is raised.

**Arguments**:

- `message` _Event_ - The event to enqueue.
- `step` _str | None_ - Optional step name to target.


**Raises**:

- `WorkflowRuntimeError` - If the target step does not exist or does not
  accept the event type.


**Examples**:

  It's common to use this method to fan-out events:

```python
@step
async def my_step(self, ctx: Context, ev: StartEvent) -> WorkerEvent | GatherEvent:
    for i in range(10):
        ctx.send_event(WorkerEvent(msg=i))
    return GatherEvent()
```

  You also see this method used from the caller side to send events into the workflow:

```python
handler = my_workflow.run(...)
async for ev in handler.stream_events():
    if isinstance(ev, SomeEvent):
        handler.ctx.send_event(SomeOtherEvent(msg="Hello!"))

result = await handler
```

<a id="workflows.context.context.Context.wait_for_event"></a>

#### wait\_for\_event

```python
async def wait_for_event(event_type: Type[T],
                         waiter_event: Event | None = None,
                         waiter_id: str | None = None,
                         requirements: dict[str, Any] | None = None,
                         timeout: float | None = 2000) -> T
```

Wait for the next matching event of type `event_type`.

Optionally emits a `waiter_event` to the event stream once per `waiter_id` to
inform callers that the workflow is waiting for external input.
This helps to prevent duplicate waiter events from being sent to the event stream.

**Arguments**:

- `event_type` _type[T]_ - Concrete event class to wait for.
- `waiter_event` _Event | None_ - Optional event to write to the stream
  once when the wait begins.
- `waiter_id` _str | None_ - Stable identifier to avoid emitting multiple
  waiter events for the same logical wait.
- `requirements` _dict[str, Any] | None_ - Key/value filters that must be
  satisfied by the event via `event.get(key) == value`.
- `timeout` _float | None_ - Max seconds to wait. `None` means no
  timeout. Defaults to 2000 seconds.


**Returns**:

- `T` - The received event instance of the requested type.


**Raises**:

- `asyncio.TimeoutError` - If the timeout elapses.


**Examples**:

```python
@step
async def my_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
    response = await ctx.wait_for_event(
        HumanResponseEvent,
        waiter_event=InputRequiredEvent(msg="What's your name?"),
        waiter_id="user_name",
        timeout=60,
    )
    return StopEvent(result=response.response)
```

<a id="workflows.context.context.Context.write_event_to_stream"></a>

#### write\_event\_to\_stream

```python
def write_event_to_stream(ev: Event | None) -> None
```

Enqueue an event for streaming to [WorkflowHandler]](workflows.handler.WorkflowHandler).

**Arguments**:

- `ev` _Event | None_ - The event to stream. `None` can be used as a
  sentinel in some streaming modes.


**Examples**:

```python
@step
async def my_step(self, ctx: Context, ev: StartEvent) -> StopEvent:
    ctx.write_event_to_stream(ev)
    return StopEvent(result="ok")
```

<a id="workflows.context.context.Context.get_result"></a>

#### get\_result

```python
def get_result() -> RunResultT
```

Return the final result of the workflow run.

**Examples**:

```python
result = await my_workflow.run(..., ctx=ctx)
result_agent = ctx.get_result()
```


**Returns**:

- `RunResultT` - The value provided via a `StopEvent`.

<a id="workflows.context.context.Context.streaming_queue"></a>

#### streaming\_queue

```python
@property
def streaming_queue() -> asyncio.Queue
```

The internal queue used for streaming events to callers.

<a id="workflows.context.context.Context.shutdown"></a>

#### shutdown

```python
async def shutdown() -> None
```

Shut down the workflow run and clean up background tasks.

Cancels all outstanding workers, waits for them to finish, and marks the
context as not running. Queues and state remain available so callers can
inspect or drain leftover events.

<a id="workflows.context.context.Context.add_step_worker"></a>

#### add\_step\_worker

```python
def add_step_worker(name: str, step: Callable, config: StepConfig,
                    verbose: bool, run_id: str,
                    resource_manager: ResourceManager) -> None
```

Spawn a background worker task to process events for a step.

**Arguments**:

- `name` _str_ - Step name.
- `step` _Callable_ - Step function (sync or async).
- `config` _StepConfig_ - Resolved configuration for the step.
- `verbose` _bool_ - If True, print step activity.
- `run_id` _str_ - Run identifier for instrumentation.
- `resource_manager` _ResourceManager_ - Resource injector for the step.

<a id="workflows.context.context.Context.add_cancel_worker"></a>

#### add\_cancel\_worker

```python
def add_cancel_worker() -> None
```

Install a worker that turns a cancel flag into an exception.

When the cancel flag is set, a `WorkflowCancelledByUser` will be raised
internally to abort the run.

**See Also**:

  - [WorkflowCancelledByUser](#workflows.errors.WorkflowCancelledByUser)

<a id="workflows.context.utils.get_qualified_name"></a>

#### get\_qualified\_name

```python
def get_qualified_name(value: Any) -> str
```

Get the qualified name of a value.

**Arguments**:

- `value` _Any_ - The value to get the qualified name for.


**Returns**:

- `str` - The qualified name in the format 'module.class'.


**Raises**:

- `AttributeError` - If value does not have __module__ or __class__ attributes

<a id="workflows.context.utils.import_module_from_qualified_name"></a>

#### import\_module\_from\_qualified\_name

```python
def import_module_from_qualified_name(qualified_name: str) -> Any
```

Import a module from a qualified name.

**Arguments**:

- `qualified_name` _str_ - The fully qualified name of the module to import.


**Returns**:

- `Any` - The imported module object.


**Raises**:

- `ValueError` - If qualified_name is empty or malformed
- `ImportError` - If module cannot be imported
- `AttributeError` - If attribute cannot be found in module

<a id="workflows.handler.WorkflowHandler"></a>

## WorkflowHandler

```python
class WorkflowHandler(asyncio.Future[RunResultT])
```

Handle a running workflow: await results, stream events, access context, or cancel.

Instances are returned by [Workflow.run](#workflows.workflow.Workflow.run).
They can be awaited for the final result and support streaming intermediate
events via [stream_events](#workflows.handler.WorkflowHandler.stream_events).

**See Also**:

  - [Context](#workflows.context.context.Context)
  - [StopEvent](#workflows.events.StopEvent)

<a id="workflows.handler.WorkflowHandler.ctx"></a>

#### ctx

```python
@property
def ctx() -> Context | None
```

The workflow [Context](#workflows.context.context.Context) for this run.

<a id="workflows.handler.WorkflowHandler.is_done"></a>

#### is\_done

```python
def is_done() -> bool
```

Return True when the workflow has completed.

<a id="workflows.handler.WorkflowHandler.stream_events"></a>

#### stream\_events

```python
async def stream_events() -> AsyncGenerator[Event, None]
```

Stream events from the workflow execution as they occur.

This method provides real-time access to events generated during workflow
execution, allowing for monitoring and processing of intermediate results.
Events are yielded in the order they are generated by the workflow.

The stream includes all events written to the context's streaming queue,
and terminates when a [StopEvent](#workflows.events.StopEvent) is
encountered, indicating the workflow has completed.

**Returns**:

  AsyncGenerator[Event, None]: An async generator that yields Event objects
  as they are produced by the workflow.


**Raises**:

- `ValueError` - If the context is not set on the handler.
- `WorkflowRuntimeError` - If all events have already been consumed by a
  previous call to `stream_events()` on the same handler instance.


**Examples**:

```python
handler = workflow.run()

# Stream and process events in real-time
async for event in handler.stream_events():
    if isinstance(event, StopEvent):
        print(f"Workflow completed with result: {event.result}")
    else:
        print(f"Received event: {event}")

# Get final result
result = await handler
```


**Notes**:

  Events can only be streamed once per handler instance. Subsequent
  calls to `stream_events()` will raise a WorkflowRuntimeError.

<a id="workflows.handler.WorkflowHandler.cancel_run"></a>

#### cancel\_run

```python
async def cancel_run() -> None
```

Cancel the running workflow.

Signals the underlying context to raise
[WorkflowCancelledByUser](#workflows.errors.WorkflowCancelledByUser),
which will be caught by the workflow and gracefully end the run.

**Examples**:

```python
handler = workflow.run()
await handler.cancel_run()
```

<a id="workflows.events.DictLikeModel"></a>

## DictLikeModel

```python
class DictLikeModel(BaseModel)
```

Base Pydantic model class that mimics a dict interface for dynamic fields.

Known, typed fields behave like regular Pydantic attributes. Any extra
keyword arguments are stored in an internal dict and can be accessed through
both attribute and mapping semantics. This hybrid model enables flexible
event payloads while preserving validation for declared fields.

PrivateAttr:
    _data (dict[str, Any]): Underlying Python dict for dynamic fields.

<a id="workflows.events.DictLikeModel.__init__"></a>

#### \_\_init\_\_

```python
def __init__(**params: Any)
```

Class constructor.

NOTE: fields and private_attrs are pulled from params by name.

<a id="workflows.events.DictLikeModel.__bool__"></a>

#### \_\_bool\_\_

```python
def __bool__() -> bool
```

Make test `if event:` pass on Event instances.

<a id="workflows.events.Event"></a>

## Event

```python
class Event(DictLikeModel)
```

Base class for all workflow events.

Events are light-weight, serializable payloads passed between steps.
They support both attribute and mapping access to dynamic fields.

**Examples**:

  Subclassing with typed fields:

```python
from pydantic import Field

class CustomEv(Event):
    score: int = Field(ge=0)

e = CustomEv(score=10)
print(e.score)
```


**See Also**:

  - [StartEvent](#workflows.events.StartEvent)
  - [StopEvent](#workflows.events.StopEvent)
  - [InputRequiredEvent](#workflows.events.InputRequiredEvent)
  - [HumanResponseEvent](#workflows.events.HumanResponseEvent)

<a id="workflows.events.StartEvent"></a>

## StartEvent

```python
class StartEvent(Event)
```

Implicit entry event sent to kick off a `Workflow.run()`.

<a id="workflows.events.StopEvent"></a>

## StopEvent

```python
class StopEvent(Event)
```

Terminal event that signals the workflow has completed.

The `result` property contains the return value of the workflow run. When a
custom stop event subclass is used, the workflow result is that event
instance itself.

**Examples**:

```python
# default stop event: result holds the value
return StopEvent(result={"answer": 42})
```

  Subclassing to provide a custom result:

```python
class MyStopEv(StopEvent):
    pass

@step
async def my_step(self, ctx: Context, ev: StartEvent) -> MyStopEv:
    return MyStopEv(result={"answer": 42})
```

<a id="workflows.events.InputRequiredEvent"></a>

## InputRequiredEvent

```python
class InputRequiredEvent(Event)
```

Emitted when human input is required to proceed.

Automatically written to the event stream if returned from a step.

If returned from a step, it does not need to be consumed by other steps and will pass validation.
It's expected that the caller will respond to this event and send back a [HumanResponseEvent](#workflows.events.HumanResponseEvent).

Use this directly or subclass it.

Typical flow: a step returns `InputRequiredEvent`, callers consume it from
the stream and send back a [HumanResponseEvent](#workflows.events.HumanResponseEvent).

**Examples**:

```python
from workflows.events import InputRequiredEvent, HumanResponseEvent

class HITLWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> InputRequiredEvent:
        return InputRequiredEvent(prefix="What's your name? ")

    @step
    async def my_step(self, ev: HumanResponseEvent) -> StopEvent:
        return StopEvent(result=ev.response)
```

<a id="workflows.events.HumanResponseEvent"></a>

## HumanResponseEvent

```python
class HumanResponseEvent(Event)
```

Carries a human's response for a prior input request.

If consumed by a step and not returned by another, it will still pass validation.

**Examples**:

```python
from workflows.events import InputRequiredEvent, HumanResponseEvent

class HITLWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> InputRequiredEvent:
        return InputRequiredEvent(prefix="What's your name? ")

    @step
    async def my_step(self, ev: HumanResponseEvent) -> StopEvent:
        return StopEvent(result=ev.response)
```

<a id="workflows.resource._Resource"></a>

## \_Resource

```python
class _Resource(Generic[T])
```

Internal wrapper for resource factories.

Wraps sync/async factories and records metadata such as the qualified name
and cache behavior.

<a id="workflows.resource._Resource.call"></a>

#### call

```python
async def call() -> T
```

Invoke the underlying factory, awaiting if necessary.

<a id="workflows.resource.ResourceDefinition"></a>

## ResourceDefinition

```python
class ResourceDefinition(BaseModel)
```

Definition for a resource injection requested by a step signature.

**Attributes**:

- `name` _str_ - Parameter name in the step function.
- `resource` __Resource_ - Factory wrapper used by the manager to produce the dependency.

<a id="workflows.resource.Resource"></a>

#### Resource

```python
def Resource(factory: Callable[..., T], cache: bool = True) -> _Resource[T]
```

Declare a resource to inject into step functions.

**Arguments**:

- `factory` _Callable[..., T]_ - Function returning the resource instance. May be async.
- `cache` _bool_ - If True, reuse the produced resource across steps. Defaults to True.


**Returns**:

- `_Resource[T]` - A resource descriptor to be used in `typing.Annotated`.


**Examples**:

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

<a id="workflows.resource.ResourceManager"></a>

## ResourceManager

```python
class ResourceManager()
```

Manage resource lifecycles and caching across workflow steps.

<a id="workflows.resource.ResourceManager.set"></a>

#### set

```python
async def set(name: str, val: Any) -> None
```

Register a resource instance under a name.

<a id="workflows.resource.ResourceManager.get"></a>

#### get

```python
async def get(resource: _Resource) -> Any
```

Return a resource instance, honoring cache settings.

<a id="workflows.resource.ResourceManager.get_all"></a>

#### get\_all

```python
def get_all() -> dict[str, Any]
```

Return all materialized resources.

<a id="workflows.server.server.WorkflowServer"></a>

## WorkflowServer

```python
class WorkflowServer()
```

A server that exposes workflows as a REST API.

This class provides a Starlette-based web server to manage and run `Workflow`
objects. It exposes endpoints to list available workflows, run them
synchronously or asynchronously, retrieve results, and stream events.

**Arguments**:

- `middleware` _list[Middleware] | None_ - A list of Starlette middleware to
  be applied to the application. If None, a default CORS middleware
  allowing all origins, methods, and headers is used.

<a id="workflows.server.server.WorkflowServer.serve"></a>

#### serve

```python
async def serve(host: str = "localhost",
                port: int = 80,
                uvicorn_config: dict[str, Any] | None = None) -> None
```

Run the server.

<a id="workflows.server.utils.nanoid"></a>

#### nanoid

```python
def nanoid(size: int = 10) -> str
```

Returns a unique identifier with the format 'kY2xP9hTnQ'.

<a id="workflows.types.RunResultT"></a>

#### RunResultT

Type aliases for workflow results.

- `StopEventT`: Generic bound to [StopEvent](#workflows.events.StopEvent)
- `RunResultT`: Result type returned by a workflow run. Today it allows either
  a `StopEventT` subclass or `Any` for backward compatibility; future versions
  may restrict this to `StopEventT` only.

<a id="workflows.utils.StepSignatureSpec"></a>

## StepSignatureSpec

```python
class StepSignatureSpec(BaseModel)
```

A Pydantic model representing the signature of a step function or method.

<a id="workflows.utils.inspect_signature"></a>

#### inspect\_signature

```python
def inspect_signature(fn: Callable) -> StepSignatureSpec
```

Given a function, ensure the signature is compatible with a workflow step.

**Arguments**:

- `fn` _Callable_ - The function to inspect.


**Returns**:

- `StepSignatureSpec` - A specification object containing:
  - `accepted_events` - Dictionary mapping parameter names to their event types
  - `return_types` - List of return type annotations
  - `context_parameter` - Name of the context parameter if present


**Raises**:

- `TypeError` - If fn is not a callable object

<a id="workflows.utils.validate_step_signature"></a>

#### validate\_step\_signature

```python
def validate_step_signature(spec: StepSignatureSpec) -> None
```

Validate that a step signature specification meets workflow requirements.

**Arguments**:

- `spec` _StepSignatureSpec_ - The signature specification to validate.


**Raises**:

- `WorkflowValidationError` - If the signature is invalid for a workflow step.

<a id="workflows.utils.get_steps_from_class"></a>

#### get\_steps\_from\_class

```python
def get_steps_from_class(_class: object) -> dict[str, Callable]
```

Given a class, return the list of its methods that were defined as steps.

**Arguments**:

- `_class` _object_ - The class to inspect for step methods.


**Returns**:

  dict[str, Callable]: A dictionary mapping step names to their corresponding methods.

<a id="workflows.utils.get_steps_from_instance"></a>

#### get\_steps\_from\_instance

```python
def get_steps_from_instance(workflow: object) -> dict[str, Callable]
```

Given a workflow instance, return the list of its methods that were defined as steps.

**Arguments**:

- `workflow` _object_ - The workflow instance to inspect.


**Returns**:

  dict[str, Callable]: A dictionary mapping step names to their corresponding methods.

<a id="workflows.utils.is_free_function"></a>

#### is\_free\_function

```python
def is_free_function(qualname: str) -> bool
```

Determines whether a certain qualified name points to a free function.

A free function is either a module-level function or a nested function.
This implementation follows PEP-3155 for handling nested function detection.

**Arguments**:

- `qualname` _str_ - The qualified name to analyze.


**Returns**:

- `bool` - True if the name represents a free function, False otherwise.


**Raises**:

- `ValueError` - If the qualified name is empty.

<a id="workflows.retry_policy.RetryPolicy"></a>

## RetryPolicy

```python
@runtime_checkable
class RetryPolicy(Protocol)
```

Policy interface to control step retries after failures.

Implementations decide whether to retry and how long to wait before the next
attempt based on elapsed time, number of attempts, and the last error.

**See Also**:

  - [ConstantDelayRetryPolicy](#workflows.retry_policy.ConstantDelayRetryPolicy)
  - [step](#workflows.decorators.step)

<a id="workflows.retry_policy.RetryPolicy.next"></a>

#### next

```python
def next(elapsed_time: float, attempts: int, error: Exception) -> float | None
```

Decide if another retry should occur and the delay before it.

**Arguments**:

- `elapsed_time` _float_ - Seconds since the first failure.
- `attempts` _int_ - Number of attempts made so far.
- `error` _Exception_ - The last exception encountered.


**Returns**:

  float | None: Seconds to wait before retrying, or `None` to stop.

<a id="workflows.retry_policy.ConstantDelayRetryPolicy"></a>

## ConstantDelayRetryPolicy

```python
class ConstantDelayRetryPolicy()
```

Retry at a fixed interval up to a maximum number of attempts.

**Examples**:

```python
@step(retry_policy=ConstantDelayRetryPolicy(delay=5, maximum_attempts=10))
async def flaky(self, ev: StartEvent) -> StopEvent:
    ...
```

<a id="workflows.retry_policy.ConstantDelayRetryPolicy.__init__"></a>

#### \_\_init\_\_

```python
def __init__(maximum_attempts: int = 3, delay: float = 5) -> None
```

Initialize the policy.

**Arguments**:

- `maximum_attempts` _int_ - Maximum consecutive attempts. Defaults to 3.
- `delay` _float_ - Seconds to wait between attempts. Defaults to 5.

<a id="workflows.retry_policy.ConstantDelayRetryPolicy.next"></a>

#### next

```python
def next(elapsed_time: float, attempts: int, error: Exception) -> float | None
```

Return the fixed delay while attempts remain; otherwise `None`.

<a id="workflows.errors.WorkflowValidationError"></a>

## WorkflowValidationError

```python
class WorkflowValidationError(Exception)
```

Raised when the workflow configuration or step signatures are invalid.

<a id="workflows.errors.WorkflowTimeoutError"></a>

## WorkflowTimeoutError

```python
class WorkflowTimeoutError(Exception)
```

Raised when a workflow run exceeds the configured timeout.

<a id="workflows.errors.WorkflowRuntimeError"></a>

## WorkflowRuntimeError

```python
class WorkflowRuntimeError(Exception)
```

Raised for runtime errors during step execution or event routing.

<a id="workflows.errors.WorkflowDone"></a>

## WorkflowDone

```python
class WorkflowDone(Exception)
```

Internal control-flow exception used to terminate workers at run end.

<a id="workflows.errors.WorkflowCancelledByUser"></a>

## WorkflowCancelledByUser

```python
class WorkflowCancelledByUser(Exception)
```

Raised when a run is cancelled via the handler or programmatically.

<a id="workflows.errors.WorkflowStepDoesNotExistError"></a>

## WorkflowStepDoesNotExistError

```python
class WorkflowStepDoesNotExistError(Exception)
```

Raised when addressing a step that does not exist in the workflow.

<a id="workflows.errors.WorkflowConfigurationError"></a>

## WorkflowConfigurationError

```python
class WorkflowConfigurationError(Exception)
```

Raised when a logical configuration error is detected pre-run.

<a id="workflows.errors.ContextSerdeError"></a>

## ContextSerdeError

```python
class ContextSerdeError(Exception)
```

Raised when serializing/deserializing a `Context` fails.

<a id="workflows.workflow.Workflow"></a>

## Workflow

```python
class Workflow(metaclass=WorkflowMeta)
```

Event-driven orchestrator to define and run application flows using typed steps.

A `Workflow` is composed of `@step`-decorated callables that accept and emit
typed [Event](#workflows.events.Event)s. Steps can be declared as instance
methods or as free functions registered via the decorator.

Key features:
- Validation of step signatures and event graph before running
- Typed start/stop events
- Streaming of intermediate events
- Optional human-in-the-loop events
- Retry policies per step
- Resource injection

**Examples**:

  Basic usage:

```python
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent

class MyFlow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")

result = await MyFlow(timeout=60).run(topic="Pirates")
```

  Custom start/stop events and streaming:

```python
handler = MyFlow().run()
async for ev in handler.stream_events():
    ...
result = await handler
```


**See Also**:

  - [step](#workflows.decorators.step)
  - [Event](#workflows.events.Event)
  - [Context](#workflows.context.context.Context)
  - [WorkflowHandler](#workflows.handler.WorkflowHandler)
  - [RetryPolicy](#workflows.retry_policy.RetryPolicy)

<a id="workflows.workflow.Workflow.__init__"></a>

#### \_\_init\_\_

```python
def __init__(timeout: float | None = 45.0,
             disable_validation: bool = False,
             verbose: bool = False,
             resource_manager: ResourceManager | None = None,
             num_concurrent_runs: int | None = None) -> None
```

Initialize a workflow instance.

**Arguments**:

- `timeout` _float | None_ - Max seconds to wait for completion. `None`
  disables the timeout.
- `disable_validation` _bool_ - Skip pre-run validation of the event graph
  (not recommended).
- `verbose` _bool_ - If True, print step activity.
- `resource_manager` _ResourceManager | None_ - Custom resource manager
  for dependency injection.
- `num_concurrent_runs` _int | None_ - Limit on concurrent `run()` calls.

<a id="workflows.workflow.Workflow.start_event_class"></a>

#### start\_event\_class

```python
@property
def start_event_class() -> type[StartEvent]
```

The `StartEvent` subclass accepted by this workflow.

Determined by inspecting step input types.

<a id="workflows.workflow.Workflow.stop_event_class"></a>

#### stop\_event\_class

```python
@property
def stop_event_class() -> type[RunResultT]
```

The `StopEvent` subclass produced by this workflow.

Determined by inspecting step return annotations.

<a id="workflows.workflow.Workflow.add_step"></a>

#### add\_step

```python
@classmethod
def add_step(cls, func: Callable) -> None
```

Adds a free function as step for this workflow instance.

It raises an exception if a step with the same name was already added to the workflow.

<a id="workflows.workflow.Workflow.run"></a>

#### run

```python
@dispatcher.span
def run(ctx: Context | None = None,
        start_event: StartEvent | None = None,
        **kwargs: Any) -> WorkflowHandler
```

Run the workflow and return a handler for results and streaming.

This schedules the workflow execution in the background and returns a
[WorkflowHandler](#workflows.handler.WorkflowHandler) that can be awaited
for the final result or used to stream intermediate events.

You may pass either a concrete `start_event` instance or keyword
arguments that will be used to construct the inferred
[StartEvent](#workflows.events.StartEvent) subclass.

**Arguments**:

- `ctx` _Context | None_ - Optional context to resume or share state
  across runs. If omitted, a fresh context is created.
- `start_event` _StartEvent | None_ - Optional explicit start event.
- `**kwargs` _Any_ - Keyword args to initialize the start event when
  `start_event` is not provided.


**Returns**:

- `WorkflowHandler` - A future-like object to await the final result and
  stream events.


**Raises**:

- `WorkflowValidationError` - If validation fails and validation is
  enabled.
- `WorkflowRuntimeError` - If the start event cannot be created from kwargs.
- `WorkflowTimeoutError` - If execution exceeds the configured timeout.


**Examples**:

```python
# Create and run with kwargs
handler = MyFlow().run(topic="Pirates")

# Stream events
async for ev in handler.stream_events():
    ...

# Await final result
result = await handler
```

  If you subclassed the start event, you can also directly pass it in:

```python
result = await my_workflow.run(start_event=MyStartEvent(topic="Pirates"))
```

<a id="workflows.decorators.step"></a>

#### step

```python
def step(*args: Any,
         workflow: Type["Workflow"] | None = None,
         num_workers: int = 4,
         retry_policy: RetryPolicy | None = None) -> Callable
```

Decorate a callable to declare it as a workflow step.

The decorator inspects the function signature to infer the accepted event
type, return event types, optional `Context` parameter (optionally with a
typed state model), and any resource injections via `typing.Annotated`.

When applied to free functions, provide the workflow class via
`workflow=MyWorkflow`. For instance methods, the association is automatic.

**Arguments**:

- `workflow` _type[Workflow] | None_ - Workflow class to attach the free
  function step to. Not required for methods.
- `num_workers` _int_ - Number of workers for this step. Defaults to 4.
- `retry_policy` _RetryPolicy | None_ - Optional retry policy for failures.


**Returns**:

- `Callable` - The original function, annotated with internal step metadata.


**Raises**:

- `WorkflowValidationError` - If signature validation fails or when decorating
  a free function without specifying `workflow`.


**Examples**:

  Method step:

```python
class MyFlow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="done")
```

  Free function step:

```python
class MyWorkflow(Workflow):
    pass

@step(workflow=MyWorkflow)
async def generate(ev: StartEvent) -> NextEvent: ...
```
