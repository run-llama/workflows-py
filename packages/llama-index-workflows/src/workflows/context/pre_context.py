"""Pre-run context - configuration face before workflow execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from pydantic import ValidationError

from workflows.context.context_types import MODEL_T, SerializedContext
from workflows.context.serializers import BaseSerializer, JsonSerializer
from workflows.errors import ContextSerdeError
from workflows.runtime.types.internal_state import BrokerState

if TYPE_CHECKING:
    from workflows.workflow import Workflow


class PreContext(Generic[MODEL_T]):
    """Context state before workflow starts.

    Provides access to workflow configuration and serialization
    for persistence/restoration. State store is created by the runtime.
    """

    _init_snapshot: SerializedContext
    _serializer: BaseSerializer

    def __init__(
        self,
        workflow: "Workflow",
        previous_context: dict[str, Any] | None = None,
        serializer: BaseSerializer | None = None,
    ) -> None:
        self._serializer = serializer or JsonSerializer()

        # Parse the serialized context
        if previous_context is not None:
            try:
                # Auto-detect and convert V0 to V1 if needed
                previous_context_parsed = SerializedContext.from_dict_auto(
                    previous_context
                )
                # Validate it fully parses synchronously to avoid delayed validation errors
                BrokerState.from_serialized(
                    previous_context_parsed, workflow, self._serializer
                )
            except ValidationError as e:
                raise ContextSerdeError(
                    f"Context dict specified in an invalid format: {e}"
                ) from e
        else:
            previous_context_parsed = SerializedContext()

        self._init_snapshot = previous_context_parsed

    @property
    def is_running(self) -> bool:
        """Whether the workflow is currently running.

        Returns the is_running state from the init snapshot, which may be True
        if restoring a context that was previously mid-run.
        """
        return self._init_snapshot.is_running

    @property
    def init_snapshot(self) -> SerializedContext:
        """The initial serialized context snapshot."""
        return self._init_snapshot

    @property
    def serializer(self) -> BaseSerializer:
        """The serializer used for this context."""
        return self._serializer
