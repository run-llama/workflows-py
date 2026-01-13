from typing import Dict, List

from workflows.server.abstract_workflow_store import (
    AbstractWorkflowStore,
    HandlerQuery,
    PersistentHandler,
)


def _matches_query(handler: PersistentHandler, query: HandlerQuery) -> bool:
    # Empty lists should match nothing (short-circuit)
    if query.handler_id_in is not None:
        if len(query.handler_id_in) == 0:
            return False
        if handler.handler_id not in query.handler_id_in:
            return False

    if query.workflow_name_in is not None:
        if len(query.workflow_name_in) == 0:
            return False
        if handler.workflow_name not in query.workflow_name_in:
            return False

    if query.status_in is not None:
        if len(query.status_in) == 0:
            return False
        if handler.status not in query.status_in:
            return False

    if query.is_idle is not None:
        handler_is_idle = handler.idle_since is not None
        if query.is_idle != handler_is_idle:
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

    async def delete(self, query: HandlerQuery) -> int:
        to_delete = [
            handler_id
            for handler_id, handler in list(self.handlers.items())
            if _matches_query(handler, query)
        ]
        for handler_id in to_delete:
            del self.handlers[handler_id]
        return len(to_delete)
