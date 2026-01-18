# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Workflow tracker for runtime instance registration."""

from __future__ import annotations

from workflows.runtime.types.plugin import RegisteredWorkflow
from workflows.workflow import Workflow


class WorkflowTracker:
    """
    Tracks workflow instances registered with a runtime.

    Used by runtimes to collect workflows before launch() and to
    look up registered workflows at execution time.

    Uses strong references to ensure workflows survive until explicitly cleared.
    """

    def __init__(self) -> None:
        # Workflows registered before launch (strong refs to survive GC)
        self._pending: list[Workflow] = []
        self._pending_set: set[int] = set()  # Track by id for dedup
        # Names assigned to workflows (for stable DBOS registration)
        self._names: dict[int, str] = {}  # keyed by id(workflow)
        # Registered workflows after launch (control loop + steps)
        self._registered: dict[int, RegisteredWorkflow] = {}  # keyed by id(workflow)
        self._launched: bool = False

    def add(self, workflow: Workflow, name: str | None = None) -> None:
        """Add a workflow to be registered at launch time."""
        if self._launched:
            raise RuntimeError(
                "Cannot add workflows after launch(). "
                "Create workflows before calling runtime.launch()."
            )
        wf_id = id(workflow)
        if wf_id not in self._pending_set:
            self._pending.append(workflow)
            self._pending_set.add(wf_id)
        if name is not None:
            self._names[wf_id] = name

    def remove(self, workflow: Workflow) -> None:
        """Remove a workflow from pending registration."""
        wf_id = id(workflow)
        self._pending = [wf for wf in self._pending if id(wf) != wf_id]
        self._pending_set.discard(wf_id)
        self._names.pop(wf_id, None)

    def get_name(self, workflow: Workflow) -> str | None:
        """Get the assigned name for a workflow, if any."""
        return self._names.get(id(workflow))

    def get_pending(self) -> list[Workflow]:
        """Get all pending workflows."""
        return list(self._pending)

    def mark_launched(self) -> None:
        """Mark that launch() has been called."""
        self._launched = True

    @property
    def is_launched(self) -> bool:
        """Whether launch() has been called."""
        return self._launched

    def set_registered(
        self, workflow: Workflow, registered: RegisteredWorkflow
    ) -> None:
        """Store the registered workflow (wrapped control loop + steps)."""
        self._registered[id(workflow)] = registered

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        """Get the registered workflow if available."""
        return self._registered.get(id(workflow))

    def clear(self) -> None:
        """Clear all tracking state (for destroy())."""
        self._pending.clear()
        self._pending_set.clear()
        self._names.clear()
        self._registered.clear()
        self._launched = False
