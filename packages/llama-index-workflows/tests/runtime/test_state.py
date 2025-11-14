from json import JSONDecodeError
import pytest
from workflows.context.context_types import SerializedContext, SerializedContextV0
from workflows.workflow import Workflow


def test_deserialize_broken_state_raises_validation_error(workflow: Workflow) -> None:
    """Test that broken V0 state raises an error when deserializing."""
    broken_state = {
        "state": {},
        "streaming_queue": "[]",
        "queues": {"middle_step": "not-deserializable-as-a-queue"},
        "event_buffers": {},
        "in_progress": {},
        "accepted_events": [],
        "broker_log": [],
        "is_running": True,
        "waiting_ids": [],
    }

    # This is V0 format (no version field)
    serialized_v0 = SerializedContextV0.model_validate(broken_state)

    # The broken queue string should cause an error during V0->V1 conversion
    # because the queue value is not valid JSON
    with pytest.raises(JSONDecodeError):
        SerializedContext.from_v0(serialized_v0)
