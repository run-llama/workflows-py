from enum import Enum


class PostEventsHandlerIdResponse200Status(str, Enum):
    SENT = "sent"

    def __str__(self) -> str:
        return str(self.value)
