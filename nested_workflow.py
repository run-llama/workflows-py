"""
Example demonstrating nested workflow composition.

This example shows how workflows can be composed together, with child workflow
steps "unrolled" into the parent workflow, sharing the same Context.
"""

import asyncio
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent


# ============================================================================
# Child Workflow - A reusable text processing workflow
# ============================================================================


class ProcessTextEvent(StartEvent):
    """Start event for the text processing workflow."""

    text: str


class TextProcessedEvent(StopEvent):
    """Stop event with processed text result."""

    processed_text: str


class UppercaseEvent(Event):
    """Intermediate event after uppercasing."""

    text: str


class TextProcessingWorkflow(Workflow):
    """A simple workflow that processes text through multiple steps."""

    @step
    async def start_processing(self, ev: ProcessTextEvent) -> UppercaseEvent:
        """First step: convert to uppercase."""
        print(f"[TextProcessing] Converting to uppercase: {ev.text}")
        return UppercaseEvent(text=ev.text.upper())

    @step
    async def add_decoration(self, ev: UppercaseEvent) -> TextProcessedEvent:
        """Second step: add decoration."""
        decorated = f"*** {ev.text} ***"
        print(f"[TextProcessing] Adding decoration: {decorated}")
        return TextProcessedEvent(processed_text=decorated)


# ============================================================================
# Parent Workflow - Uses the child workflow as a component
# ============================================================================


class MainStartEvent(StartEvent):
    """Start event for the main workflow."""

    input_text: str
    repeat_count: int = Field(default=1)


class TriggerTextProcessingEvent(Event):
    """Event that triggers the nested text processing workflow."""

    text: str


class TextProcessingCompleteEvent(Event):
    """Event emitted when text processing completes."""

    processed_text: str


class FinalResultEvent(StopEvent):
    """Final result of the main workflow."""

    final_text: str
    total_length: int


class MainWorkflowState(BaseModel):
    """State for the main workflow."""

    processing_count: int = Field(default=0)
    repeat_count: int = Field(default=1)
    original_input: str = Field(default="")


class MainWorkflow(Workflow):
    """
    Main workflow that includes the TextProcessingWorkflow.

    This demonstrates how a child workflow's steps can be "unrolled" into
    the parent, with proper event mapping at the boundaries.
    """

    @step
    async def start(
        self, ctx: Context[MainWorkflowState], ev: MainStartEvent
    ) -> TriggerTextProcessingEvent:
        """Start the workflow by triggering text processing."""
        print(f"\n[Main] Starting workflow with input: {ev.input_text}")
        print(f"[Main] Will process {ev.repeat_count} time(s)")

        # Store the repeat count in state
        async with ctx.store.edit_state() as state:
            state.repeat_count = ev.repeat_count
            state.original_input = ev.input_text

        # Trigger the nested workflow
        return TriggerTextProcessingEvent(text=ev.input_text)

    @step
    async def handle_processed_text(
        self, ctx: Context[MainWorkflowState], ev: TextProcessingCompleteEvent
    ) -> TriggerTextProcessingEvent | FinalResultEvent:
        """Handle the result from nested workflow."""
        print(f"\n[Main] Received processed text: {ev.processed_text}")

        # Update state
        async with ctx.store.edit_state() as state:
            state.processing_count += 1

        # Check if we should process again
        state = await ctx.store.get_state()
        current_count = state.processing_count
        repeat_count = state.repeat_count
        original_input = state.original_input

        if current_count < repeat_count:
            print(f"[Main] Processing again ({current_count}/{repeat_count})")
            # Trigger another round of processing
            return TriggerTextProcessingEvent(
                text=f"{original_input} (round {current_count + 1})"
            )
        else:
            # We're done
            print(f"[Main] Processing complete!")
            return FinalResultEvent(
                final_text=ev.processed_text, total_length=len(ev.processed_text)
            )


# ============================================================================
# Example usage
# ============================================================================


async def main() -> None:
    """Run examples of the nested workflow."""

    print("=" * 70)
    print("Example 1: Single processing pass")
    print("=" * 70)

    workflow1 = MainWorkflow(timeout=10)

    # Include the child workflow with event mapping
    # This "unrolls" all of TextProcessingWorkflow's steps into this workflo
    workflow1.include_workflow(
        workflow_class=TextProcessingWorkflow,
        # Map our TriggerTextProcessingEvent to child's ProcessTextEvent
        start_event=TriggerTextProcessingEvent,
        # Map child's TextProcessedEvent to our TextProcessingCompleteEvent
        stop_event=TextProcessingCompleteEvent,
        # Optional: namespace to avoid step name conflicts
        namespace="text_processor",
    )

    ctx1 = Context(workflow1)
    result1 = await workflow1.run(ctx=ctx1, input_text="hello world", repeat_count=1)

    print(f"\nFinal result: {result1.final_text}")
    print(f"Total length: {result1.total_length}")

    print("\n" + "=" * 70)
    print("Example 2: Multiple processing passes")
    print("=" * 70)

    workflow2 = MainWorkflow(timeout=10)
    workflow2.include_workflow(
        workflow_class=TextProcessingWorkflow,
        start_event=TriggerTextProcessingEvent,
        stop_event=TextProcessingCompleteEvent,
        namespace="text_processor",
    )

    ctx2 = Context(workflow2)
    result2 = await workflow2.run(
        ctx=ctx2, input_text="nested workflows", repeat_count=3
    )

    print(f"\nFinal result: {result2.final_text}")
    print(f"Total length: {result2.total_length}")

    print("\n" + "=" * 70)
    print("Example 3: Running child workflow standalone")
    print("=" * 70)

    # The child workflow can also be run independently
    child_workflow = TextProcessingWorkflow(timeout=10)
    child_result = await child_workflow.run(text="standalone execution")

    print(f"\nStandalone result: {child_result.processed_text}")


if __name__ == "__main__":
    asyncio.run(main())
