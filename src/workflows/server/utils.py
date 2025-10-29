# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import secrets
import string

from workflows.protocol import HandlerData
from workflows.server.abstract_workflow_store import PersistentHandler

alphabet = string.ascii_letters + string.digits  # A-Z, a-z, 0-9


def nanoid(size: int = 10) -> str:
    """Returns a unique identifier with the format 'kY2xP9hTnQ'."""
    return "".join(secrets.choice(alphabet) for _ in range(size))


def handler_data_from_persistent(handler: PersistentHandler) -> HandlerData:
    return HandlerData(
        handler_id=handler.handler_id,
        workflow_name=handler.workflow_name,
        run_id=handler.run_id,
        status=handler.status,
        started_at=handler.started_at.isoformat() if handler.started_at else "",
        updated_at=handler.updated_at.isoformat() if handler.updated_at else None,
        completed_at=handler.completed_at.isoformat() if handler.completed_at else None,
        error=handler.error,
        result=handler.result,
    )
