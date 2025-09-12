from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Literal, Optional, List, Any
from dataclasses import dataclass
from pydantic import BaseModel


Status = Literal["running", "completed", "failed"]


@dataclass()
class HandlerQuery:
    # Matches if any of the handler_ids match
    handler_id_in: Optional[List[str]] = None
    # Matches if any of the workflow_names match
    workflow_name_in: Optional[List[str]] = None
    # Matches if the status flag matches
    status_in: Optional[List[Status]] = None


class PersistentHandler(BaseModel):
    handler_id: str
    workflow_name: str
    status: Status
    ctx: dict[str, Any]


class AbstractWorkflowStore(ABC):
    @abstractmethod
    async def query(self, query: HandlerQuery) -> List[PersistentHandler]: ...

    @abstractmethod
    async def update(self, handler: PersistentHandler) -> None: ...


class EmptyWorkflowStore(AbstractWorkflowStore):
    async def query(self, query: HandlerQuery) -> List[PersistentHandler]:
        return []

    async def update(self, handler: PersistentHandler) -> None:
        pass
