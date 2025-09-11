from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Optional, List, Any
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass()
class HandlerQuery:
    handler_id: Optional[List[str]] = None
    workflow_name: Optional[List[str]] = None
    completed: Optional[bool] = None


class PersistentHandler(BaseModel):
    handler_id: str
    workflow_name: str
    completed: bool = False
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
