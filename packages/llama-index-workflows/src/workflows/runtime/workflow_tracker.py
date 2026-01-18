# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Workflow tracker for plugin instance registration."""

from __future__ import annotations

from weakref import WeakSet

from workflows.runtime.types._identity_weak_ref import IdentityWeakKeyDict
from workflows.runtime.types.plugin import RegisteredWorkflow
from workflows.workflow import Workflow


class WorkflowTracker:
    """
    Tracks workflow instances registered with a plugin.

    Used by plugins to collect workflows before launch() and to
    look up registered workflows at runtime.
    """

    def __init__(self) -> None:
        # Workflows registered before launch (weak refs to allow GC)
        self._pending: WeakSet[Workflow] = WeakSet()
        # Names assigned to workflows (for stable DBOS registration)
        self._names: IdentityWeakKeyDict[Workflow, str] = IdentityWeakKeyDict()
        # Registered workflows after launch (control loop + steps)
        self._registered: IdentityWeakKeyDict[Workflow, RegisteredWorkflow] = (
            IdentityWeakKeyDict()
        )
        self._launched: bool = False

    def add(self, workflow: Workflow, name: str | None = None) -> None:
        """Add a workflow to be registered at launch time."""
        if self._launched:
            raise RuntimeError(
                "Cannot add workflows after launch(). "
                "Create workflows before calling plugin.launch()."
            )
        self._pending.add(workflow)
        if name is not None:
            self._names[workflow] = name

    def remove(self, workflow: Workflow) -> None:
        """Remove a workflow from pending registration."""
        self._pending.discard(workflow)
        # Note: IdentityWeakKeyDict doesn't have discard, use pop with default
        self._names.pop(workflow, None)

    def get_name(self, workflow: Workflow) -> str | None:
        """Get the assigned name for a workflow, if any."""
        return self._names.get(workflow)

    def get_pending(self) -> list[Workflow]:
        """Get all pending workflows (still alive)."""
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
        self._registered[workflow] = registered

    def get_registered(self, workflow: Workflow) -> RegisteredWorkflow | None:
        """Get the registered workflow if available."""
        return self._registered.get(workflow)

    def clear(self) -> None:
        """Clear all tracking state (for destroy())."""
        self._pending.clear()
        self._names.clear()
        self._registered.clear()
        self._launched = False
