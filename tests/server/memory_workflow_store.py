from typing import Dict, List
from workflows.server.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
)


def _matches_query(handler: PersistentHandler, query: HandlerQuery) -> bool:
    if (
        query.handler_id_in is not None
        and handler.handler_id not in query.handler_id_in
    ):
        return False
    if (
        query.workflow_name_in is not None
        and handler.workflow_name not in query.workflow_name_in
    ):
        return False
    if query.status_in is not None and handler.status not in query.status_in:
        return False
    return True


class MemoryWorkflowStore(AbstractWorkflowStore):
    def __init__(self) -> None:
        self.handlers: Dict[str, PersistentHandler] = {}

    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        return [
            handler
            for handler in self.handlers.values()
            if _matches_query(handler, query)
        ]

    async def update(self, handler: PersistentHandler) -> None:
        self.handlers[handler.handler_id] = handler
