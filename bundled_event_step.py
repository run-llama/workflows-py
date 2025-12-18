from typing import Any, ClassVar, Generic, Type, TypeVar

from workflows.decorators import make_step_function
from workflows.events import Event

from workflows import Context, Workflow, step

TInput = TypeVar("TInput", bound=Event)
TOutput = TypeVar("TOutput", bound=Event)


class BundledStep(Generic[TInput, TOutput]):
    """A reusable workflow step with bundled input/output events.

    Automatically registers itself when used as a class attribute in a Workflow.

    Subclasses should define nested Input and Output classes that inherit from Event,
    and pass them as type parameters:

        class ParseStep(BundledStep["ParseStep.Input", "ParseStep.Output"]):
            class Input(Event):
                file: bytes

            class Output(Event):
                markdown: str

            async def run_step(self, ctx: Context, ev: Input) -> Output:
                return self.Output(markdown="parsed")
    """

    # Subclasses define these as nested class attributes
    Input: ClassVar[Type[Event]]
    Output: ClassVar[Type[Event]]
    num_workers: ClassVar[int] = 4

    def __init__(self) -> None:
        self._workflow_class: Type[Workflow] | None = None
        self._attr_name: str | None = None
        self._step_func: Any = None

    def __set_name__(self, owner: Type[Workflow], name: str) -> None:
        """Called when assigned as a class attribute - auto-registers the step."""
        if issubclass(owner, Workflow):
            self._workflow_class = owner
            self._attr_name = name

            # Create and register the step function
            self._step_func = self._make_step()
            owner.add_step(self._step_func)

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Raise AttributeError so the runtime falls back to using the registered step function.

        The bundled step is registered via add_step(), and the runtime will use
        the original function when getattr raises AttributeError.
        """
        raise AttributeError(
            f"BundledStep '{self._attr_name}' is not directly accessible. "
            f"The step function is registered via Workflow._step_functions."
        )

    def _make_step(self) -> Any:
        """Create the step function with proper metadata."""
        # Get the Input/Output types from the class (not instance)
        input_type = self.__class__.Input
        output_type = self.__class__.Output

        # Capture self for the closure
        bundled_step = self

        async def step_wrapper(ctx: Context, ev: input_type) -> output_type:  # type: ignore[valid-type]
            return await bundled_step.run_step(ctx, ev)

        # Give it a unique name based on the attribute name
        step_wrapper.__name__ = self._attr_name or self.__class__.__name__
        workflow_cls = self._workflow_class
        if workflow_cls is not None:
            step_wrapper.__qualname__ = (
                f"{workflow_cls.__name__}.{step_wrapper.__name__}"
            )

        # Patch annotations so the workflow can introspect them
        step_wrapper.__annotations__ = {
            "ctx": Context,
            "ev": input_type,
            "return": output_type,
        }

        # Use make_step_function to attach metadata (same as @step decorator)
        return make_step_function(step_wrapper, num_workers=self.num_workers)

    async def run_step(self, ctx: Context, ev: TInput) -> TOutput:
        """Override this to implement step logic."""
        raise NotImplementedError


async def main() -> None:
    from workflows.events import StartEvent, StopEvent

    # Define reusable bundled step with forward-reference type parameters
    class ParseStep(BundledStep["ParseStep.Input", "ParseStep.Output"]):  # type: ignore
        class Input(Event):
            file: bytes
            tier: str
            version: str | None = None

        class Output(Event):
            markdown: str

        async def run_step(self, ctx: Context, ev: Input) -> Output:
            return ParseStep.Output(markdown="some parsed markdown")

    # Use in workflow
    class MyWorkflow(Workflow):
        parse = ParseStep()  # Auto-registers via __set_name__!

        @step
        async def start(self, ctx: Context, ev: StartEvent) -> ParseStep.Input:
            file_data = b"some file data"
            return ParseStep.Input(file=file_data, tier="enterprise")

        @step
        async def handle_result(self, ctx: Context, ev: ParseStep.Output) -> StopEvent:
            return StopEvent(result=ev.markdown)

    workflow = MyWorkflow()
    _ = await workflow.run(input_msg="Start parsing!")
    # print("Workflow result:", res)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
