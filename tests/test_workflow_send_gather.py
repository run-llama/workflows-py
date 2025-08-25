import pytest
import time

from typing import Union, Annotated, Any
from workflows.decorators import step
from workflows.events import (
    StartEvent,
    Event,
    StopEvent,
)
from workflows.context import Context
from workflows.resource import Resource
from workflows.workflow import Workflow


class InputEvent(StartEvent):
    emails: dict[str, str]


class SendEmailEvent(Event):
    contact: str
    content: str


class ProcessEmailEvent(Event):
    sent: bool
    time: float


class EmailCounter:
    def __init__(self) -> None:
        self.success = 0
        self.fail = 0

    def update(self, success: bool) -> None:
        if success:
            self.success += 1
        else:
            self.fail += 1


counter = EmailCounter()


def get_counter(*args: Any, **kwargs: Any) -> EmailCounter:
    return counter


class EmailSenderWorkflow(Workflow):
    @step
    async def collect_contacts(
        self, ev: InputEvent, ctx: Context
    ) -> Union[SendEmailEvent, None]:
        events = []
        for contact, content in ev.emails.items():
            events.append(SendEmailEvent(contact=contact, content=content))
        ctx.send_events(events)  # type: ignore
        return None

    @step
    async def process_email(
        self,
        ev: SendEmailEvent,
        ctx: Context,
        counter: Annotated[EmailCounter, Resource(get_counter)],
    ) -> ProcessEmailEvent:
        if ev.contact.startswith("j"):
            sent = False
        else:
            sent = True
        counter.update(success=sent)
        return ProcessEmailEvent(sent=sent, time=time.time())

    @step(gather=[ProcessEmailEvent])
    async def output(
        self,
        ev: ProcessEmailEvent,
        ctx: Context,
        counter: Annotated[EmailCounter, Resource(get_counter)],
    ) -> Union[StopEvent, None]:
        return StopEvent(result=f"Sent {counter.success} emails; {counter.fail} failed")


@pytest.fixture
def workflow() -> EmailSenderWorkflow:
    return EmailSenderWorkflow()


@pytest.fixture
def emails() -> dict[str, str]:
    return {
        "john.doe@example.com": "Dear John,\nHow are you?",
        "johanna.doe@example.com": "Dear Johanna,\nHow are you?",
        "mario.rossi@email.it": "Hello Mario!\nHope everything is going weel :)",
        "alice.smith@example.com": "Hi Alice, just checking in.",
        "bob.johnson@email.com": "Meeting notes for tomorrow.",
        "charlie.brown@mail.org": "Your order confirmation.",
        "david.williams@example.net": "Regarding your recent inquiry.",
        "eve.jones@email.co.uk": "Special offer just for you!",
        "frank.miller@mail.net": "Follow up on our conversation.",
        "grace.davis@example.com": "Important update about your account.",
        "heidi.moore@email.org": "Your weekly newsletter.",
        "ivan.taylor@mail.com": "A quick question for you.",
        "judy.anderson@example.net": "Upcoming event details.",
        "kevin.thomas@email.co.uk": "Your monthly statement is ready.",
        "liam.jackson@mail.net": "Exclusive discount code.",
        "mia.white@example.com": "Feedback request for your recent purchase.",
        "noah.harris@email.org": "Invitation to our webinar.",
        "olivia.martin@mail.com": "Happy birthday!",
        "peter.garcia@example.net": "Important security alert.",
        "quinn.rodriguez@email.co.uk": "Your subscription is expiring soon.",
    }


@pytest.mark.asyncio
async def test_dispatch_receive(
    workflow: EmailSenderWorkflow, emails: dict[str, str]
) -> None:
    result = await workflow.run(start_event=InputEvent(emails=emails))
    assert result == "Sent 17 emails; 3 failed"
