from typing import Literal


class StateNotFoundError(Exception):
    """Raise when state cannot be loaded from AgentCore Memory"""

    def __init__(self, cause: Literal["event_id", "data"]):
        self.cause: Literal["event_id", "data"] = cause

    def __str__(self) -> str:
        if self.cause == "event":
            return "Could not load state from AgentCore memory as no event was found corresponding to the provided ID"
        else:
            return "Could not load state from AgentCore memory as no blob data was found within the event's payload"

    def __repr__(self) -> str:
        if self.cause == "event":
            return "Could not load state from AgentCore memory as no event was found corresponding to the provided ID"
        else:
            return "Could not load state from AgentCore memory as no blob data was found within the event's payload"
