# SPDX-License-Identifier: MIT
"""AgentCore entrypoint for LlamaIndex Workflows.

Wraps workflows for AWS Bedrock AgentCore Runtime using BedrockAgentCoreApp.

Session Integration
-------------------
``context.session_id`` is used as the **default handler_id** for all workflow
operations.  This gives a natural 1:1 mapping between AgentCore sessions and
workflow handlers — re-invoking the same session returns the cached result
(completed), awaits (running), or starts fresh (failed/cancelled).

Action-Based Routing
--------------------
The entrypoint supports an ``"action"`` key in the payload to expose the full
WorkflowServer capabilities over a single AgentCore invoke channel:

    run          — run synchronously (default when action is omitted)
    run_nowait   — start without waiting, return handler_id
    get_result   — poll handler status and result
    get_events   — retrieve recorded workflow events
    send_event   — inject event into running workflow (human-in-the-loop)
    cancel       — cancel a running handler
    list_workflows — list registered workflow names
    list_handlers  — list handlers (filter by workflow/status)
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from bedrock_agentcore import BedrockAgentCoreApp
from llama_agents.appserver.workflow_loader import (
    load_environment_variables,
    load_workflows,
    validate_required_env_vars,
)
from llama_agents.client import HandlerData
from llama_agents.core.deployment_config import (
    read_deployment_config_from_git_root_or_cwd,
)
from llama_agents.server._store.abstract_workflow_store import (
    HandlerQuery,
)
from llama_agents.server._store.sqlite.sqlite_workflow_store import (
    SqliteWorkflowStore,
)
from pydantic import BaseModel, ValidationError
from workflows import Workflow
from workflows.context.serializers import JsonSerializer
from workflows.events import Event, StartEvent

from ._service import AgentCoreService

logger = logging.getLogger(__name__)
app = BedrockAgentCoreApp()

AGENTCORE_HOST = "0.0.0.0"
AGENTCORE_PORT = 8080


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class WorkflowResult(BaseModel):
    """Response for a single workflow run."""

    workflow: str
    status: Literal["completed", "failed"]
    result: str | None = None
    error: str | None = None
    session_id: str | None = None


class HandlerResult(BaseModel):
    """Response for handler-level operations."""

    handler_id: str
    session_id: str | None = None
    workflow_name: str | None = None
    run_id: str | None = None
    status: str | None = None
    result: str | None = None
    error: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


# ---------------------------------------------------------------------------
# Workflow loading
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_workflows() -> tuple[dict[str, Workflow], str, str | None]:
    config_dir = Path.cwd()
    if not (config_dir / "pyproject.toml").exists():
        raise FileNotFoundError(
            "No pyproject.toml found at "
            f"{config_dir}.\n"
            "Add a pyproject.toml to your project and re-run."
        )
    config = read_deployment_config_from_git_root_or_cwd(
        Path.cwd(), config_dir
    )  # let errors bubble up if misconfigured

    # Load env vars from the deployment config (env, env_files) before
    # importing workflow modules — mirrors appserver behaviour.
    load_environment_variables(config, config_dir)
    validate_required_env_vars(config)

    workflows = load_workflows(config)
    has_default = any(key == "default" for key in list(workflows.keys()))
    default_workflow = "default" if has_default else next(iter(workflows))
    file_workflow = None
    for name, wf in workflows.items():
        if wf.start_event_class.__name__ == "FileEvent":
            file_workflow = name
            break
    return workflows, default_workflow, file_workflow


AGENTCORE_WORKSPACE = "/mnt/workspace"
SQLITE_DB_NAME = "workflows.db"


def _get_sqlite_db_path() -> str:
    """Return the SQLite DB path, preferring AgentCore session storage."""
    workspace = Path(AGENTCORE_WORKSPACE)
    if workspace.exists() and workspace.is_dir():
        return str(workspace / SQLITE_DB_NAME)
    return SQLITE_DB_NAME


@functools.lru_cache(maxsize=1)
def get_agentcore_service() -> AgentCoreService:
    workflows, _, _ = _load_workflows()
    db_path = _get_sqlite_db_path()
    store = SqliteWorkflowStore(db_path=db_path)
    logger.info("Using SQLite workflow store at %s", db_path)
    service = AgentCoreService(app=app, store=store)
    for k in workflows:
        service.add_workflow(k, workflows[k])
    return service


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_json_serializer = JsonSerializer()


def _dt_str(dt: Any) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def _handler_to_result(
    handler: HandlerData, session_id: str | None = None
) -> dict[str, Any]:
    """Convert HandlerData to a JSON-safe response dict."""
    result = handler.result
    if result is not None:
        result = _json_serializer.serialize(result)

    return HandlerResult(
        handler_id=handler.handler_id,
        session_id=session_id,
        workflow_name=getattr(handler, "workflow_name", None),
        run_id=getattr(handler, "run_id", None),
        status=handler.status,
        result=result,
        error=getattr(handler, "error", None),
        started_at=_dt_str(getattr(handler, "started_at", None)),
        updated_at=_dt_str(getattr(handler, "updated_at", None)),
        completed_at=_dt_str(getattr(handler, "completed_at", None)),
    ).model_dump()


def _serialize_event(stored_event: Any) -> dict[str, Any]:
    """Convert a StoredEvent to a JSON-safe dict."""
    envelope = stored_event.event
    return {
        "sequence": stored_event.sequence,
        "timestamp": _dt_str(stored_event.timestamp),
        "value": envelope.value if hasattr(envelope, "value") else None,
        "type": getattr(envelope, "type", None),
        "qualified_name": getattr(envelope, "qualified_name", None),
    }


# ---------------------------------------------------------------------------
# Session / handler ID resolution
# ---------------------------------------------------------------------------


def _resolve_handler_id(payload: dict[str, Any], session_id: str) -> str:
    """Determine the handler_id to use.

    Priority:
    1. Explicit ``handler_id`` in payload (for multi-workflow sessions)
    2. ``context.session_id`` (default — one handler per session)
    """
    return payload.get("handler_id") or session_id


# ---------------------------------------------------------------------------
# Payload parsing (unchanged from original for backwards compat)
# ---------------------------------------------------------------------------


def _parse_and_validate_payload(
    workflows: dict[str, Any],
    default_workflow: str,
    file_workflow: str | None,
    payload: dict[str, Any],
) -> tuple[str, StartEvent] | tuple[str, str]:
    """Parse incoming payload to determine workflow and event data.

    Supports:
        - Explicit: {"workflow": "process-file", "start_event": {"file_id": "123"}}
        - Shorthand: {"file_id": "123"} -> routes to file-processing workflow
        - Default: {} -> routes to default workflow
    """
    workflow_name = payload.get("workflow")
    event_data = payload.get("start_event", {})

    if not workflow_name:
        if "file_id" in payload and file_workflow is not None:
            workflow_name = file_workflow
            event_data = {"file_id": payload["file_id"]}
        else:
            workflow_name = default_workflow
    if workflow_name not in workflows:
        return workflow_name, f"Workflow not found: {workflow_name}"
    wf = workflows[workflow_name]
    start_cls = wf.start_event_class
    try:
        data = start_cls.model_validate(event_data)
    except ValidationError as e:
        return workflow_name, f"Invalid input data: {e}"
    return workflow_name, data


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


async def _action_run(
    payload: dict[str, Any], session_id: str, *, nowait: bool = False
) -> dict[str, Any]:
    """Run (or resume) a workflow with session-aware handler ID."""
    workflows, default_workflow, file_workflow = _load_workflows()
    parsed = _parse_and_validate_payload(
        workflows, default_workflow, file_workflow, payload
    )

    # Validation error
    if len(parsed) == 2 and all(isinstance(p, str) for p in parsed):
        workflow_name, error = parsed[0], parsed[1]
        return WorkflowResult(
            workflow=str(workflow_name),
            error=str(error),
            status="failed",
            session_id=session_id,
        ).model_dump()

    workflow_name = str(parsed[0])
    start_event: StartEvent = parsed[1]  # type: ignore[assignment]
    handler_id = _resolve_handler_id(payload, session_id)
    service = get_agentcore_service()

    try:
        handler_data = await service.run_workflow_with_session(
            workflow_name,
            start_event,
            handler_id=handler_id,
            nowait=nowait,
        )
    except Exception as e:
        logger.error("Workflow '%s' failed: %s", workflow_name, e, exc_info=True)
        return WorkflowResult(
            workflow=workflow_name,
            error=f"Workflow failed: {e}",
            status="failed",
            session_id=session_id,
        ).model_dump()

    return _handler_to_result(handler_data, session_id)


async def _action_get_result(
    payload: dict[str, Any], session_id: str
) -> dict[str, Any]:
    """Get handler status and result."""
    handler_id = _resolve_handler_id(payload, session_id)
    service = get_agentcore_service()

    handler = await service.get_handler(handler_id)
    if handler is None:
        return {"error": f"Handler '{handler_id}' not found", "session_id": session_id}

    return _handler_to_result(handler, session_id)


async def _action_get_events(
    payload: dict[str, Any], session_id: str
) -> dict[str, Any]:
    """Retrieve recorded events for a handler."""
    handler_id = _resolve_handler_id(payload, session_id)
    service = get_agentcore_service()

    after_seq = payload.get("after_sequence", -1)
    limit = payload.get("limit", 200)
    after = after_seq if after_seq >= 0 else None

    events = await service.get_events(handler_id, after_sequence=after, limit=limit)
    return {
        "handler_id": handler_id,
        "session_id": session_id,
        "events": [_serialize_event(e) for e in events],
    }


async def _action_send_event(
    payload: dict[str, Any], session_id: str
) -> dict[str, Any]:
    """Send an event into a running workflow (human-in-the-loop)."""
    handler_id = _resolve_handler_id(payload, session_id)
    event_data = payload.get("event")
    if not event_data:
        return {"error": "event is required", "session_id": session_id}

    step = payload.get("step")
    value = event_data.get("value", event_data)
    event = Event(**value) if isinstance(value, dict) else Event()

    service = get_agentcore_service()
    await service.send_event(handler_id, event, step=step)
    return {"status": "sent", "handler_id": handler_id, "session_id": session_id}


async def _action_cancel(payload: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Cancel a running workflow handler."""
    handler_id = _resolve_handler_id(payload, session_id)
    purge = payload.get("purge", False)

    service = get_agentcore_service()
    result = await service.cancel_handler(handler_id, purge=purge)
    return {
        "status": result or "not_found",
        "handler_id": handler_id,
        "session_id": session_id,
    }


async def _action_list_workflows(
    _payload: dict[str, Any], session_id: str
) -> dict[str, Any]:
    """List registered workflow names."""
    service = get_agentcore_service()
    return {"workflows": service.get_workflow_names(), "session_id": session_id}


async def _action_list_handlers(
    payload: dict[str, Any], session_id: str
) -> dict[str, Any]:
    """List handlers, optionally filtered by workflow and/or status."""
    query_kwargs: dict[str, Any] = {}
    if payload.get("workflow"):
        query_kwargs["workflow_name_in"] = [payload["workflow"]]
    if payload.get("status"):
        query_kwargs["status_in"] = [payload["status"]]

    service = get_agentcore_service()
    handlers = await service.query_handlers(HandlerQuery(**query_kwargs))
    return {
        "handlers": [_handler_to_result(h, session_id) for h in handlers],
        "session_id": session_id,
    }


# Action dispatch table
_ACTIONS: dict[str, Any] = {
    "run": lambda p, sid: _action_run(p, sid, nowait=False),
    "run_nowait": lambda p, sid: _action_run(p, sid, nowait=True),
    "get_result": _action_get_result,
    "get_events": _action_get_events,
    "send_event": _action_send_event,
    "cancel": _action_cancel,
    "list_workflows": _action_list_workflows,
    "list_handlers": _action_list_handlers,
}


# ---------------------------------------------------------------------------
# AgentCore entrypoint
# ---------------------------------------------------------------------------


@app.entrypoint
async def invoke(payload: dict, context: Any) -> dict[str, Any]:
    """Single entrypoint that dispatches to workflow operations.

    ``context.session_id`` is used as the default handler_id, giving a
    natural 1:1 mapping between AgentCore sessions and workflow handlers.

    When ``"action"`` is omitted, falls back to synchronous run-and-wait
    for backwards compatibility.
    """
    session_id: str = getattr(context, "session_id", "")
    action = payload.get("action")

    # Default: run-and-wait (backwards compatible)
    if action is None:
        return await _action_run(payload, session_id, nowait=False)

    handler_fn = _ACTIONS.get(action)
    if handler_fn is None:
        return {
            "error": f"Unknown action '{action}'",
            "available": list(_ACTIONS.keys()),
            "session_id": session_id,
        }

    try:
        return await handler_fn(payload, session_id)
    except Exception as e:
        logger.exception("Action '%s' failed", action)
        return {"error": str(e), "action": action, "session_id": session_id}
